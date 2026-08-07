# sam2_inference_service.py

"""
Servicio de Inferencia de SAM 2 para Predicción de Transitabilidad (Traversability Prediction).

Este módulo actúa como un envoltorio (wrapper) de alto nivel sobre la arquitectura de 
SAM 2 (Segment Anything Model). Su objetivo es procesar imágenes RGB capturadas por 
la cámara frontal de un robot o vehículo, y deducir qué porción de la imagen corresponde 
al suelo transitable (asfalto, camino seguro) frente a obstáculos.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

from .build_sam import build_sam2
from .sam2_image_predictor import SAM2ImagePredictor

class SAM_TP:
    """
    Clase principal que maneja el ciclo de vida del modelo SAM 2 para tareas de navegación.
    (SAM_TP = Segment Anything Model - Traversability Prediction).

    CONCEPTOS CLAVE:
    - Ciclo de vida en Memoria: Los modelos de IA modernos (Foundation Models) como SAM 2 
      son extremadamente pesados (ocupan Gigabytes de VRAM en la GPU). Esta clase está 
      diseñada para instanciarse una sola vez al arrancar el sistema, manteniendo los 
      pesos cargados en la memoria, permitiendo que las subsecuentes llamadas de inferencia 
      (frame a frame) sean mucho más rápidas.
    """
    def __init__(self, sam2_cfg_path, sam2_checkpoint_path, score_thresh=0.0, multimask=False):
        """
        Carga e inicializa el modelo SAM 2 en memoria exactamente una vez.

        Args:
            sam2_cfg_path (str): Ruta al archivo YAML con la configuración arquitectónica del modelo.
            sam2_checkpoint_path (str): Ruta al archivo de pesos matemáticos (.pt).
            score_thresh (float): Umbral de confianza. Puntuaciones por debajo de esto se descartan.
            multimask (bool): Si es True, permite a SAM devolver múltiples siluetas alternativas 
                              para resolver ambigüedades.
        """
        print("start")
        print(sam2_cfg_path)
        print(sam2_checkpoint_path)
        self.sam2_model = build_sam2(sam2_cfg_path, sam2_checkpoint_path)
        print("end")

        self.score_thresh = score_thresh
        self.multimask_output = multimask

    def run_sam2_inference(self, input_image_np: np.ndarray):
        """
        Ejecuta la inferencia de SAM 2 sobre un único fotograma RGB para descubrir el suelo seguro.

        CONCEPTOS CLAVE:
        - Prompting Heurístico (La regla del suelo): SAM 2 no sabe qué es "el suelo" por defecto; 
          necesita que le "hagan clic" en el objeto a segmentar. Este código asume una regla física de 
          robótica: la cámara del vehículo está apuntando hacia adelante, por lo que los píxeles ubicados 
          en el borde inferior absoluto de la imagen SIEMPRE corresponden al piso directamente debajo 
          del robot. El algoritmo hace 3 clics virtuales automáticos (izquierdo, central y derecho 
          en la parte baja) etiquetándolos como "1" (Foreground/Objeto a buscar) para decirle a la IA: 
          "Expande esta textura y dime hasta dónde llega el camino".
        - Logits en lugar de Máscaras Binarias: Al usar `return_logits=True`, la IA no devuelve solo 
          1s y 0s (Blanco/Negro). Devuelve números crudos continuos (Logits). Un logit alto significa 
          "estoy seguro de que esto es suelo", un logit negativo es "estoy seguro de que esto es pared". 
          Esto permite al planificador de rutas calcular riesgos progresivos (costos suaves).
        - Generación del Heatmap (Mapa de Calor): Convierte esos logits abstractos en una imagen térmica 
          fácil de entender para humanos mediante normalización Min-Max y aplicando la paleta de colores 'jet' 
          (donde el rojo/caliente es alto puntaje y el azul/frío es bajo puntaje).

        Args:
            input_image_np (np.ndarray): El fotograma de entrada como matriz NumPy de forma (Alto, Ancho, 3).

        Returns:
            dict: Un diccionario que contiene:
                - "heatmap": Arreglo NumPy (H, W, 3) con los colores RGB del mapa de calor visual.
                - "logits": Arreglo NumPy (H, W) continuo con las predicciones crudas de la red neuronal.
        """
        pil_image = Image.fromarray(input_image_np)
        predictor = SAM2ImagePredictor(
            sam_model=self.sam2_model,
            mask_threshold=self.score_thresh
        )
        predictor.reset_predictor()
        predictor.set_image(pil_image)

        width, height = pil_image.size
        bottom_left  = (0,        height - 1)
        bottom_right = (width-1,  height - 1)
        bottom_mid   = ((width-1)//2, height - 1)

        point_coords = np.array([bottom_left, bottom_right, bottom_mid], dtype=np.float32)
        point_labels = np.ones(len(point_coords), dtype=np.int32)  # 1=foreground

        masks, iou_predictions, low_res_logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=self.multimask_output,
            return_logits=True,
            normalize_coords=False
        )
        # Choose best
        best_mask_idx = int(iou_predictions.argmax()) if self.multimask_output else 0
        best_mask_logit = masks[best_mask_idx]  # shape(H, W) float

        # This is the raw "score map" we can interpret as cost or traversability
        score_map = best_mask_logit

        # Create color heatmap from the same array
        mini = score_map.min()
        maxi = score_map.max()
        eps = 1e-8
        normalized = (score_map - mini) / (maxi - mini + eps)
        heatmap_array = (cm.get_cmap('jet')(normalized) * 255).astype(np.uint8)  # (H,W,4)
        heatmap_array = heatmap_array[..., :3]  # drop alpha => (H,W,3) in RGB

        return {
            "heatmap": heatmap_array,
            "logits": score_map
        }