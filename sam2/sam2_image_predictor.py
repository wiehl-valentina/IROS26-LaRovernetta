# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Módulo Predictor de Imágenes para SAM 2 (Segment Anything Model 2).

Este módulo proporciona la interfaz principal para realizar la segmentación 
de imágenes estáticas. Implementa el patrón de diseño donde la "extracción 
de características" (Image Embedding) se realiza una sola vez mediante un 
backbone pesado, y luego se pueden realizar múltiples predicciones ultra-rápidas 
(Prompting) sobre esa misma imagen utilizando puntos, cajas o máscaras previas.
"""

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    """
    Clase orquestadora para la inferencia de SAM 2 en imágenes individuales o lotes (batches).
    
    CONCEPTOS CLAVE:
    - Separación de Cómputo (Backbone vs Decoder): Para lograr eficiencia en tiempo real, 
      el modelo procesa la imagen pesada una sola vez (en `set_image`) usando su Backbone 
      (generalmente un Vision Transformer tipo Hiera). Esto genera un "Embedding" (resumen 
      matemático). Luego, el usuario puede darle miles de indicaciones (clics) usando 
      `predict` y el modelo solo ejecuta el decodificador (Mask Decoder), que es instantáneo.
    """
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Inicializa el predictor envolviendo la arquitectura base de SAM 2 y configurando 
        las transformaciones y filtros de post-procesamiento.

        Args:
          sam_model (SAM2Base): La red neuronal SAM 2 instanciada y cargada en memoria.
          mask_threshold (float): Umbral de binarización. La red neuronal escupe "logits" 
            (números crudos). Todo logit mayor a este umbral (por defecto 0.0) se considerará 
            como parte de la máscara (blanco/True), y lo menor será fondo (negro/False).
          max_hole_area (int): Post-procesamiento morfológico. Si es > 0, rellenará automáticamente 
            los agujeros oscuros dentro de la máscara generada que sean menores a este tamaño en píxeles.
          max_sprinkle_area (int): Post-procesamiento morfológico. Si es > 0, eliminará el "ruido" 
            (pequeñas islas blancas desconectadas) menores a este tamaño en píxeles.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Constructor alternativo (Factory Method) para inicializar el predictor 
        descargando automáticamente los pesos desde el repositorio de Hugging Face.

        Args:
          model_id (str): El identificador oficial del modelo en Hugging Face 
                          (ej. 'facebook/sam2.1-hiera-large').
          **kwargs: Argumentos adicionales que se propagarán al constructor principal.

        Returns:
          (SAM2ImagePredictor): Una instancia lista para usarse con el modelo pre-entrenado.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Ejecuta el paso computacional más pesado: calcula los "Image Embeddings" 
        de la imagen proporcionada y los almacena en la caché interna de la clase.
        
        CONCEPTOS CLAVE:
        - Image Embedding: Es una representación tensorial comprimida de la imagen 
          que contiene toda su semántica visual. 
        - Características Multiescala (High Res Feats): SAM 2 extrae información a diferentes 
          resoluciones (256x256, 128x128, 64x64) para poder segmentar tanto objetos gigantes 
          como detalles diminutos con precisión perfecta en los bordes.

        Args:
          image (np.ndarray o PIL Image): La imagen de entrada en formato RGB. 
            Si es numpy, debe tener formato HWC (Alto, Ancho, Canales). 
            Si es PIL, la librería maneja internamente la estructura. 
            Los valores de los píxeles deben estar en el rango [0, 255].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Variante de `set_image` diseñada para procesar múltiples imágenes simultáneamente 
        (Batched Inference). Esto exprime el paralelismo de la GPU, acelerando el proceso 
        cuando se tienen varias imágenes independientes.

        Args:
          image_list (List[np.ndarray]): Lista de imágenes de entrada a embeber. 
          Todas deben estar en formato RGB y HWC (Alto, Ancho, Canales).
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Ejecuta la inferencia de máscaras sobre un lote (batch) de imágenes previamente
        cargadas con `set_image_batch`. Toma listas de indicaciones (prompts) y devuelve
        listas de resultados correspondientes a cada imagen.

        Args:
            (Comparte la misma semántica de parámetros que el método `predict`, pero
            cada parámetro es una lista donde el índice [i] corresponde a la imagen [i]).

        Returns:
            Tupla de tres listas: (Máscaras finales, Puntajes de Calidad/IoU, Máscaras de baja resolución).
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predice las máscaras en la imagen actual basándose en los 'prompts' (indicaciones) dados.

        CONCEPTOS CLAVE:
        - Prompts (Indicaciones): SAM 2 funciona adivinando lo que el usuario quiere. Puedes pedirle 
          objetos pasándole puntos específicos (clics), una caja (bounding box) que encierre el objeto, 
          o una máscara previa incompleta para que la mejore.
        - Salida Multimáscara (Ambigüedad Resolutiva): Si haces clic en la llanta de un auto, SAM no 
          sabe si quieres segmentar solo el neumático, la rueda completa, o el auto entero. Si 
          `multimask_output=True`, devolverá 3 máscaras distintas cubriendo esas 3 escalas lógicas.

        Args:
          point_coords (np.ndarray o None): Arreglo Nx2 de coordenadas (X, Y) en píxeles.
          point_labels (np.ndarray o None): Arreglo de longitud N indicando la intención del punto.
            Un '1' significa "quiero este objeto" (Foreground), un '0' significa "excluye esta zona" (Background).
          box (np.ndarray o None): Un arreglo de 4 elementos indicando una caja en formato [X_min, Y_min, X_max, Y_max].
          mask_input (np.ndarray o None): Una máscara cruda de baja resolución proveniente de una inferencia anterior.
            Sirve para refinamiento iterativo. Su forma debe ser 1xHxW (para SAM suele ser 256x256).
          multimask_output (bool): Activa la devolución de múltiples interpretaciones del objeto (3 máscaras)
            para resolver ambigüedades originadas por un solo clic. Si el prompt es muy claro (ej. una caja ajustada), 
            desactivarlo suele dar mejores resultados.
          return_logits (bool): Si es True, no aplica el umbral de binarización y devuelve los números 
            crudos matemáticos (útil para análisis de incerteza o combinaciones de máscaras).
          normalize_coords (bool): Si es True, asume que los puntos proporcionados ya están escalados a las 
            dimensiones reales de la imagen y los normaliza internamente.

        Returns:
          (np.ndarray): Las máscaras finales con forma CxHxW (C = cantidad de máscaras, H y W = tamaño original de la imagen).
          (np.ndarray): Arreglo 1D de longitud C con el puntaje IoU predictivo (la propia red evalúa qué tan buena cree que es su máscara).
          (np.ndarray): Arreglo de logits de baja resolución (Cx256x256). Pueden inyectarse nuevamente en el parámetro `mask_input` 
            en la siguiente llamada para refinar el resultado.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        """
        Función interna de acondicionamiento de prompts. Convierte los arreglos crudos de NumPy 
        en tensores de PyTorch, los traslada al hardware adecuado (GPU/CPU) y aplica las 
        transformaciones geométricas necesarias para que coincidan con la resolución interna del modelo.
        """

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        El núcleo del motor de inferencia (Decoder). 
        Toma el embedding de la imagen pre-calculado y los prompts (indicaciones) ya acondicionados,
        y los pasa por el "Prompt Encoder" y el "Mask Decoder" de SAM 2 para generar las siluetas finales.

        CONCEPTOS CLAVE:
        - Sparse vs Dense Prompts: Puntos y cajas son 'Sparse' (dispersos, solo unas pocas coordenadas).
          Las máscaras previas son 'Dense' (densas, una matriz completa de píxeles). SAM 2 codifica 
          ambos por separado y luego los combina para instruir a la red neuronal de qué debe dibujar.
        - Escalado Inverso (Upscale): La red neuronal escupe máscaras pequeñas (usualmente 256x256) 
          por eficiencia térmica y de memoria. Al final del método (`postprocess_masks`), estas 
          máscaras son estiradas de nuevo a la altísima resolución de la imagen fotográfica original.

        Args:
            (Parámetros internos tensores equivalentes a la función pública `predict`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Los tensores de las máscaras generadas, 
            los puntajes de calidad IoU, y las máscaras de baja resolución para futuros ciclos.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Retorna el tensor que contiene los Image Embeddings calculados y cacheados 
        para la imagen actualmente establecida en el predictor.
        
        Su forma topológica (shape) típica es 1xCxHxW, donde C suele ser 256 dimensiones 
        de profundidad conceptual, y H=W=64 es la dimensión espacial reducida de SAM.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        """
        Propiedad de conveniencia que devuelve el hardware actual de procesamiento 
        asignado al modelo (ej. 'cuda:0' para tarjetas NVIDIA, 'mps' para Apple Silicon o 'cpu').
        """
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Limpia la memoria del predictor. Borra los embeddings y resetea los estados booleanos 
        para dejar el entorno limpio antes de procesar una nueva imagen de origen.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False