# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Módulo de Construcción e Inicialización (Factory) para SAM 2.

Este archivo contiene las funciones principales para instanciar los modelos 
de Segmentación de Imágenes y Video (Segment Anything Model 2). 
Se encarga de fusionar las configuraciones YAML, descargar pesos pre-entrenados 
desde Hugging Face y ensamblar la arquitectura neuronal correcta.
"""

import logging
import os

import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


"""
Diccionario que mapea los identificadores oficiales de los modelos en Hugging Face (HF)
con los nombres de sus respectivos archivos de configuración YAML y archivos de pesos (.pt).
Esto permite descargar y armar el modelo correcto automáticamente con solo pasar el ID.
"""
HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """
    Construye e inicializa el modelo base de SAM 2 para segmentación de imágenes estáticas.

    CONCEPTOS CLAVE:
    - Hydra & OmegaConf: Son frameworks avanzados de configuración usados por Meta. En lugar de
      escribir un código gigante para instanciar clases, Hydra lee un archivo YAML que describe
      el "árbol" de objetos de la red neuronal y lo construye dinámicamente (`instantiate`).
    - Post-procesamiento (Multimask dinámico): SAM 2 está diseñado para manejar ambigüedades 
      (ej. si haces clic en la rueda de un auto, ¿quieres segmentar la rueda o todo el auto?). 
      Si `apply_postprocessing` es True, se inyectan reglas en Hydra para que el modelo evalúe 
      la "estabilidad" de su predicción. Si la máscara principal es inestable, automáticamente 
      activa la salida multi-máscara para ofrecer alternativas.

    Args:
        config_file (str): Ruta absoluta o relativa al archivo YAML de configuración del modelo.
        ckpt_path (str, opcional): Ruta al archivo de pesos (.pt). Si es None, carga el modelo vacío.
        device (str): Dispositivo de cómputo destino ('cuda', 'cpu', 'mps').
        mode (str): Modo de PyTorch ('eval' para inferencia, 'train' para entrenamiento).
        hydra_overrides_extra (list): Lista de cadenas para sobrescribir parámetros específicos del YAML.
        apply_postprocessing (bool): Si es True, inyecta las reglas de estabilidad multi-máscara.

    Returns:
        torch.nn.Module: El modelo SAM 2 completamente inicializado y cargado en el dispositivo.
    """

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    config_file = os.path.abspath(config_file)
    config_dir = os.path.dirname(config_file)
    config_name = os.path.basename(config_file)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, job_name="sam2_job"):
        cfg = compose(config_name=config_name, overrides=hydra_overrides_extra)
        OmegaConf.resolve(cfg)

    # Read config and init model
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    """
    Construye e inicializa el modelo predictivo de SAM 2 específico para procesamiento de VIDEO.

    CONCEPTOS CLAVE:
    - SAM2VideoPredictor vs SAM2Base: A diferencia del modelo de imágenes sueltas, el predictor de video
      incluye un "Banco de Memoria" (Memory Bank). Esto le permite rastrear objetos a lo largo del tiempo,
      recordando cómo se veía el objeto en fotogramas anteriores.
    - VOS (Video Object Segmentation): Si `vos_optimized` es True, se sobrescribe la clase objetivo (target)
      hacia `SAM2VideoPredictorVOS`. Esta versión compila el encoder de imágenes de la red para que 
      sea muchísimo más rápido al procesar flujos secuenciales de fotogramas.
    - Binarización y Relleno de agujeros: Las directivas de Hydra inyectadas aquí aseguran que lo que el
      usuario marca con sus clics se memorice de forma estricta (binarize_mask), y rellena automáticamente
      pequeños agujeros en las máscaras de baja resolución para evitar que el rastreo pierda solidez con el tiempo.

    Returns:
        SAM2VideoPredictor: La arquitectura optimizada para video lista para inferencia.
    """
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    """
    Función auxiliar para conectarse a los servidores de Hugging Face y descargar los archivos 
    necesarios de forma transparente para el usuario.
    
    Busca el `model_id` en el diccionario global `HF_MODEL_ID_TO_FILENAMES`, descarga el archivo
    de pesos (o usa el que ya esté en caché local) y devuelve las rutas listas para instanciar.
    """
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    """
    Envoltorio (Wrapper) conveniente para instanciar el modelo de imágenes usando 
    únicamente el identificador de Hugging Face (ej. "facebook/sam2.1-hiera-large").
    Descarga los pesos y llama a la función principal `build_sam2`.
    """
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    """
    Envoltorio (Wrapper) conveniente para instanciar el modelo predictivo de video usando 
    únicamente el identificador de Hugging Face.
    Descarga los pesos y llama a la función principal `build_sam2_video_predictor`.
    """
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    """
    Carga los pesos matemáticos pre-entrenados (.pt) dentro de la arquitectura de la red neuronal.

    CONCEPTOS CLAVE:
    - weights_only=True: Es una medida crítica de ciberseguridad en PyTorch. Previene que archivos 
      maliciosos inyecten y ejecuten código arbitrario en Python al ser cargados, asegurando que 
      solo se extraigan los tensores numéricos.
    - strict matching: Comprueba meticulosamente que cada capa declarada en la arquitectura del 
      modelo coincida exactamente con las capas guardadas en el archivo de pesos. Si sobran 
      o faltan claves (missing_keys / unexpected_keys), aborta la ejecución para evitar 
      comportamientos erráticos silenciosos.
    """
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")