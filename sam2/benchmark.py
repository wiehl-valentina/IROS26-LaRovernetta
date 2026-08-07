# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
BENCHMARK DE SEGMENTACIÓN DE VIDEO CON SAM 2 (Segment Anything Model 2)
=============================================================================
Este script es una herramienta de evaluación de rendimiento (Benchmarking) 
diseñada para medir los Fotogramas Por Segundo (FPS) que puede procesar 
el modelo SAM 2 al rastrear un objeto a través de un video completo.

CONCEPTOS CLAVE DEL PIPELINE:
1. Optimización de Hardware: Configura la GPU para usar matemática de precisión mixta.
2. Inicialización de Estado: Pre-procesa todos los fotogramas del video en memoria.
3. Prompting: Recibe un "clic" del usuario en el primer fotograma para saber qué rastrear.
4. Propagación (VOS): Rastrea el objeto a lo largo del tiempo usando un mecanismo de atención (Memory Bank).
=============================================================================
"""

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

"""
-----------------------------------------------------------------------------
FASE 1: ACELERACIÓN DE HARDWARE Y PRECISIÓN MIXTA
-----------------------------------------------------------------------------
SAM 2 es un modelo extremadamente pesado. Para lograr procesar video en 
tiempo real, este bloque exprime al máximo las capacidades de las GPUs modernas.

CONCEPTOS CLAVE:
- torch.autocast (bfloat16): "Brain Floating Point". Es un formato de número 
  decimal de 16 bits creado por Google. Ocupa la mitad de RAM de video (VRAM) 
  que el formato estándar (float32) y acelera la matemática neuronal, manteniendo 
  el mismo rango dinámico para evitar que los gradientes o activaciones exploten.
- TF32 (TensorFloat-32): Si tienes una GPU NVIDIA arquitectura Ampere (RTX Serie 3000 
  en adelante), esta configuración enciende los "Tensor Cores" especiales del chip. 
  Realiza las multiplicaciones de matrices internamente a menor precisión pero 
  devuelve el resultado en 32 bits, acelerando enormemente la inferencia sin 
  pérdida de calidad visible.
-----------------------------------------------------------------------------
"""
# Only cuda supported
assert torch.cuda.is_available()
device = torch.device("cuda")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

"""
-----------------------------------------------------------------------------
FASE 2: CARGA DEL MODELO BASE
-----------------------------------------------------------------------------
CONCEPTOS CLAVE:
- vos_optimized=True: VOS significa "Video Object Segmentation". A diferencia de 
  la segmentación de imágenes sueltas, SAM 2 en modo video mantiene un "Banco de Memoria" 
  (Memory Bank). Esta bandera activa optimizaciones internas específicas para streaming 
  y gestión de colas de memoria, ideal para inferencia secuencial.
-----------------------------------------------------------------------------
"""
# Config and checkpoint
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Build video predictor with vos_optimized=True setting
predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device, vos_optimized=True
)


"""
-----------------------------------------------------------------------------
FASE 3: PRE-PROCESAMIENTO E INICIALIZACIÓN DEL ESTADO DE VIDEO
-----------------------------------------------------------------------------
CONCEPTOS CLAVE:
- predictor.init_state(): Este es el paso más costoso. La red neuronal extrae los 
  "Image Embeddings" (la representación matemática de los píxeles) de TODOS los 
  fotogramas del directorio y los guarda en la memoria (inference_state). 
  Sin esto, el modelo no tendría el contexto necesario para saber cómo cambia 
  la iluminación o el fondo a lo largo del video.
-----------------------------------------------------------------------------
"""
# Initialize with video
video_dir = "notebooks/videos/bedroom"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)


"""
-----------------------------------------------------------------------------
FASE 4: CONFIGURACIÓN DEL BENCHMARK Y EL PROMPT
-----------------------------------------------------------------------------
CONCEPTOS CLAVE:
- Warm-up (Calentamiento): En CUDA (GPU), la primera vez que ejecutas un modelo 
  tarda mucho más porque tiene que asignar memoria, inicializar núcleos y cargar 
  el caché. Un benchmark honesto siempre descarta las primeras X iteraciones (warm_up) 
  para medir la velocidad "crucero" real de la GPU.
- Puntos y Etiquetas (Prompting): Le decimos a SAM 2 exactamente QUÉ objeto rastrear 
  en el fotograma 0 (ann_frame_idx). 
  * points: Coordenada X=210, Y=350.
  * labels: `1` significa "clic positivo" (este es el objeto). Un `0` significaría 
    "clic negativo" (excluir esta zona).
-----------------------------------------------------------------------------
"""
# Number of runs, warmup etc
warm_up, runs = 5, 25
verbose = True
num_frames = len(frame_names)
total, count = 0, 0
torch.cuda.empty_cache()

# We will select an object with a click.
# See video_predictor_example.ipynb for more detailed explanation
ann_frame_idx, ann_obj_id = 0, 1
# Add a positive click at (x, y) = (210, 350)
# For labels, `1` means positive click
points = np.array([[210, 350]], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

"""
-----------------------------------------------------------------------------
FASE 5: BUCLE DE PROPAGACIÓN TEMPORAL (INFERENCIA PURA)
-----------------------------------------------------------------------------
CONCEPTOS CLAVE:
- torch.inference_mode(): Es una versión aún más estricta y rápida de torch.no_grad(). 
  Apaga por completo cualquier rastreo de historial de versiones en los tensores. 
  Es el estándar de oro en PyTorch para obtener el máximo rendimiento al momento 
  de poner un modelo en producción (despliegue).
- propagate_in_video(): El motor real del modelo. Toma la máscara generada 
  en el fotograma 0 a raíz del clic, y usa capas de atención cruzada (Cross-Attention) 
  para buscar ese mismo objeto en el fotograma 1, luego usa la información del 0 y 1 
  para buscarlo en el 2, y así sucesivamente (Tracking).
-----------------------------------------------------------------------------
"""
# Warmup and then average FPS over several runs
with torch.autocast("cuda", torch.bfloat16):
    with torch.inference_mode():
        for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
            start = time.time()
            # Start tracking
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(inference_state):
                pass

            end = time.time()
            total += end - start
            count += 1
            if i == warm_up - 1:
                print("Warmup FPS: ", count * num_frames / total)
                total = 0
                count = 0

print("FPS: ", count * num_frames / total)