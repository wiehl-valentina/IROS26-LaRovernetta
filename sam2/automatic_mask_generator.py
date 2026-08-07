# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SAM2AutomaticMaskGenerator:
    """
    Clase principal que envuelve el modelo SAM 2 (Segment Anything Model 2) para realizar
    la segmentación automática y exhaustiva de toda una imagen sin necesidad de indicaciones
    manuales previas (Zero-shot segmentation).

    CONCEPTOS CLAVE DE ARQUITECTURA:
    - Generación por grilla (Grid Prompting): Como SAM requiere un "prompt" (un punto o caja) para saber qué
      segmentar, este generador crea automáticamente una matriz gigante de puntos esparcidos por toda
      la imagen y se los alimenta al modelo en ráfagas (batches).
    - Supresión de No Máximos (NMS): Como muchos puntos de la grilla caerán sobre el mismo objeto, 
      SAM generará múltiples máscaras idénticas (o casi idénticas). NMS se encarga de comparar cuánto
      se superponen (IoU - Intersección sobre Unión) y elimina los duplicados.
    - Estrategia de Recortes (Cropping): Para imágenes de muy alta resolución, los objetos pequeños
      pueden perderse. Esta clase puede dividir la imagen en "recortes" (crops) más pequeños, correr
      el modelo en cada uno para detectar el máximo detalle, y luego volver a unir todo.
    """
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = True,
        **kwargs,
    ) -> None:
        """
        Inicializa el generador automático de máscaras usando un modelo base SAM 2.
        Por defecto, las configuraciones están optimizadas para arquitecturas tipo HieraL.

        Args:
          model (SAM2Base): La red neuronal SAM 2 cargada en memoria y lista para predecir.
          points_per_side (int o None): Define la resolución de la grilla. Si es 32, generará 
            32x32 (1024) puntos distribuidos equitativamente sobre la imagen. Excluyente con 'point_grids'.
          points_per_batch (int): Cuántos puntos procesar en paralelo en la GPU. Subirlo acelera 
            el proceso pero consume exponencialmente más memoria VRAM.
          pred_iou_thresh (float): [Filtro de Calidad] Umbral (0 a 1). SAM predice qué tan buena cree
            que quedó la máscara generada; si su propia confianza es menor a este valor, se descarta.
          stability_score_thresh (float): [Filtro de Estabilidad] Mide cuánto cambia la máscara si
            modificamos ligeramente el umbral de binarización. Si cambia mucho, es inestable y se descarta.
          stability_score_offset (float): Factor de alteración usado para calcular la estabilidad.
          mask_threshold (float): Valor de corte para convertir los logits (salida cruda) en blanco/negro (máscara).
          box_nms_thresh (float): Umbral de NMS para descartar máscaras duplicadas en la misma zona.
          crop_n_layers (int): Si es > 0, ejecuta un análisis recursivo. El nivel 0 es la imagen entera, 
            el nivel 1 parte la imagen en 4, el nivel 2 en 16, procesando cada recorte de forma independiente.
          crop_nms_thresh (float): NMS específico para fusionar máscaras que se encimaron entre diferentes recortes.
          crop_overlap_ratio (float): Fracción de superposición entre los recortes (evita que un objeto
            quede cortado por la mitad y no sea detectado).
          crop_n_points_downscale_factor (int): Reduce la cantidad de puntos de la grilla en las capas 
            de recortes más profundas (ya que son imágenes más chicas).
          point_grids (list(np.ndarray)): Permite pasar una grilla de puntos personalizada (coordenadas 0.0 a 1.0).
          min_mask_region_area (int): Post-procesamiento. Elimina cualquier "basura" visual (islas de 
            píxeles sueltos) menores a esta cantidad de área. Requiere OpenCV.
          output_mode (str): Formato de entrega. Puede ser 'binary_mask' (matriz 2D real), 'uncompressed_rle' o 
            'coco_rle' (Run-Length Encoding, hiper-comprimido, ideal para millones de máscaras).
          use_m2m (bool): Mask-to-Mask (M2M). Ejecuta un ciclo extra de refinamiento metiéndole a SAM
            su propia máscara anterior como pista visual.
          multimask_output (bool): Permite que de un solo punto salgan hasta 3 máscaras (ej. parte, objeto entero, conjunto).
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Constructor alternativo que permite inicializar la clase descargando directamente
        los pesos del modelo desde el repositorio de Hugging Face.

        Args:
          model_id (str): Identificador oficial en Hugging Face (ej. "facebook/sam2-hiera-large").
          **kwargs: Parámetros adicionales a inyectar en la creación de la clase.

        Returns:
          Una instancia totalmente configurada de SAM2AutomaticMaskGenerator.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Ejecuta la inferencia completa sobre una imagen para encontrar todos los objetos posibles.

        CONCEPTOS CLAVE:
        - @torch.no_grad(): Desactiva el cálculo de gradientes. Esto es esencial para inferencia
          porque ahorra masivamente memoria RAM y ciclos de CPU/GPU (no estamos entrenando a la IA, 
          solo pidiéndole predicciones).

        Args:
          image (np.ndarray): Imagen fuente, debe estar en formato HWC (Alto, Ancho, Canales) y 
                              con valores uint8 (0 a 255).

        Returns:
           Una lista donde cada elemento es un diccionario que representa UN objeto detectado. Contiene:
               - segmentation: La máscara física (como matriz binaria 2D o en formato comprimido RLE).
               - bbox: [x_min, y_min, ancho, alto] del objeto delimitador.
               - area: Cantidad total de píxeles que conforman el objeto.
               - predicted_iou: Puntaje de confianza reportado por la propia red neuronal.
               - point_coords: El punto (prompt) específico de la grilla que activó esta máscara.
               - stability_score: Puntaje de solidez matemática de la máscara.
               - crop_box: El marco de recorte (si se activó crop_n_layers) donde fue descubierta.
        """

        # 1. Fase de Inferencia Masiva
        mask_data = self._generate_masks(image)

        # 2. Codificación según el output_mode solicitado por el usuario
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # 3. Empaquetado en el formato estandarizado final (JSON-friendly)
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        """
        Controlador interno que gestiona la lógica de recortes (Crops).
        Si `crop_n_layers` es mayor a 0, troza la imagen y llama al motor predictivo 
        varias veces, para luego fusionar todo eliminando solapamientos.
        """
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterar sobre todos los recortes (o solo uno si es la imagen entera)
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Si hubo múltiples recortes, se deben eliminar las máscaras duplicadas
        # que surgieron en las zonas de borde/superposición.
        if len(crop_boxes) > 1:
            # Estrategia de NMS: Se le da prioridad (mayor puntaje artificial) a las máscaras
            # que provienen de recortes más pequeños, ya que capturan mejor el detalle fino.
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories (todas clase 0)
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        """
        Ejecuta todo el ciclo de predicción para un recorte particular de la imagen.

        CONCEPTOS CLAVE:
        - Image Embedding: Al llamar a `self.predictor.set_image`, SAM calcula una única vez
          el pesado "Image Embedding" (la representación neuronal de los píxeles) de este recorte.
          Luego, puede procesar miles de puntos de prompt casi instantáneamente sobre ese embedding.
        """
        # Extraer el recorte físico de la imagen
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Escalar la grilla de puntos base [0, 1] al tamaño real del recorte en píxeles
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Para no desbordar la memoria VRAM de la GPU, dividimos los miles de puntos
        # en ráfagas más manejables (batches).
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        
        # Libera el caché del embedding de la GPU
        self.predictor.reset_predictor()

        # Filtrar duplicados intra-recorte (Ej. si dos puntos cercanos cayeron sobre el mismo auto)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Devolver las coordenadas relativas del recorte al marco espacial absoluto de la imagen original
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        """
        El núcleo matemático más profundo. Inyecta los puntos en la red neuronal de SAM, 
        extrae los tensores de predicción, y aplica el filtrado de calidad y estabilidad.

        CONCEPTOS CLAVE:
        - M2M Refinement (Refinamiento Máscara a Máscara): SAM admite recibir como pista 
          no solo puntos, sino también máscaras previas en baja resolución (`low_res_masks`). 
          Si `use_m2m` está activo, toma las primeras predicciones crudas y las vuelve a pasar
          por el modelo para que las perfeccione.
        """
        orig_h, orig_w = orig_size

        # Acondicionar los tensores y subirlos al dispositivo (GPU/MPS/CPU)
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        # in_labels = 1 significa que le decimos a SAM: "Este punto pertenece a un objeto, búscalo"
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        
        # INFERENCIA: La red neuronal hace su trabajo
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # Filtrado 1: Por la confianza teórica que reportó la propia IA (IoU Predictivo)
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Filtrado 2: Por Estabilidad Matemática. Perturba el umbral y chequea si la máscara sobrevive intacta
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # Ciclo iterativo de refinamiento M2M
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            # Repite los mismos filtrados posteriores
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Convierte los tensores continuos a máscaras estrictamente Booleanas (Blanco y Negro)
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filtrar objetos artificialmente cortados porque cayeron justo en la línea de división del crop
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compresión extrema de memoria: Re-escalar al tamaño original y convertir a algoritmo Run-Length (RLE)
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Algoritmo de limpieza visual final (Postprocesamiento).
        Elimina regiones pequeñas desconectadas (basura/islas) y rellena agujeros internos 
        en las máscaras, y luego vuelve a aplicar NMS para erradicar cualquier duplicado 
        nuevo que haya surgido producto de esa limpieza.

        Modifica el objeto `mask_data` in-place y depende del motor OpenCV.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            # Eliminar vacíos dentro del objeto principal (ej. un agujero detectado erróneamente en el capó del auto)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            # Eliminar satélites desprendidos del objeto (ej. reflejos sueltos)
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            
            # Truco de puntaje para NMS: Le da puntaje 0 a las máscaras que tuvieron que ser alteradas 
            # y 1 a las "perfectas", obligando al NMS a preferir y conservar la versión más limpia.
            scores.append(float(unchanged))

        # Recalcular las cajas limitadoras (Bounding Boxes) ya que el tamaño del objeto pudo cambiar al borrar islas
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Para ahorrar CPU pesada, solo re-calcula la compresión RLE compleja en aquellas máscaras 
        # que fueron seleccionadas y efectivamente alteradas.
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]
                
        mask_data.filter(keep_by_nms)

        return mask_data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch):
        """
        Utilidad para inyectarle de nuevo la máscara anterior a la IA como prompt contextual.
        Fuerza a 'multimask_output=False' para exigirle a la red que elija la mejor silueta única posible.
        """
        new_masks = []
        new_iou_preds = []

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            best_masks, best_iou_preds, _ = self.predictor._predict(
                cur_points[:, None, :],
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
        
        masks = torch.cat(new_masks, dim=0)
        return masks, torch.cat(new_iou_preds, dim=0)