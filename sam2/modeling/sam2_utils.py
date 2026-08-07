# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
SAM 2 - FUNCIONES UTILITARIAS Y DE EVALUACIÓN (sam2_utils.py)
=============================================================================
Este archivo contiene herramientas accesorias, funciones matemáticas críticas 
y algoritmos de simulación. Muchas de estas funciones son el "pegamento" que 
hace que la arquitectura neuronal funcione, o herramientas para entrenar al 
modelo simulando clics de usuarios humanos.
=============================================================================
"""

import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Selecciona hasta `max_cond_frame_num` fotogramas de condicionamiento (donde 
    el usuario hizo clics) que estén temporalmente más cerca del fotograma actual 
    (`frame_idx`).
    
    CONCEPTOS CLAVE:
    En videos largos con mucha interacción, guardar todos los fotogramas clickeados
    saturaría la VRAM de la GPU. Esta función garantiza "Localidad Temporal":
    si estoy en el frame 50, me importa más lo que el usuario corrigió en el 
    frame 49 que lo que hizo en el frame 1.
    
    Estrategia de selección:
    - a) Obligatoriamente el frame condicionado más cercano HACIA ATRÁS (pasado).
    - b) Obligatoriamente el frame condicionado más cercano HACIA ADELANTE (futuro, si existe).
    - c) Rellena el resto del cupo con los que tengan menor distancia absoluta |t - frame_idx|.

    Outputs:
    - selected_outputs: Diccionario con los frames elegidos y sus memorias.
    - unselected_outputs: Los frames que quedaron fuera del límite.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Genera un Positional Embedding 1D absoluto usando ondas sinusoidales, exactamente
    como se describe en el paper fundacional "Attention Is All You Need" (2017).
    Se usa para darle al modelo un sentido de la "distancia temporal" (qué frame 
    vino primero y cuál después).
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """
    Patrón Factory simple. Devuelve la función matemática de activación no-lineal 
    en base a un string de configuración.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    """
    Clona de forma profunda (Deep Copy) una capa neuronal `N` veces para crear 
    una pila (stack) de capas idénticas (como las capas de un Transformer).
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    """
    Implementa "Stochastic Depth" (Profundidad Estocástica).
    Durante el entrenamiento, apaga (hace 0) ramificaciones enteras de la red neuronal 
    con una probabilidad `drop_prob`.
    Es una técnica brutal de regularización que evita que la red colapse o 
    memorice los datos, forzándola a aprender redundancias (similar al Dropout, 
    pero en lugar de neuronas sueltas, apaga bloques enteros).
    """
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        # Escala el tensor restante para que el valor "promedio" de energía 
        # fluya igual aunque se hayan apagado caminos.
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    """
    Multi-Layer Perceptron. La red neuronal "clásica" de capas lineales apiladas.
    Se utiliza generalmente al final del modelo (Head) para proyectar los embeddings 
    complejos en decisiones simples (ej. transformar el Puntero a logits de puntuación).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    """
    Layer Normalization adaptado a tensores de imagen 2D (C, H, W).
    En lugar de normalizar a través del Batch (como BatchNorm), normaliza a través 
    de los Canales (C). Esto es crítico en modelos donde el tamaño de batch es muy 
    pequeño (ej. Batch = 1), donde BatchNorm fallaría catastróficamente.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Toma una máscara real (Ground Truth) y extrae su caja delimitadora (Bounding Box), 
    añadiéndole "ruido" simulado (pequeñas alteraciones).
    
    CONCEPTOS CLAVES:
    Esta función se usa en la fase de ENTRENAMIENTO. Simula cómo un humano real dibujaría 
    una caja en la pantalla con el ratón: de forma imperfecta. Inyectar ruido previene 
    que el modelo dependa de cajas matemáticas milimétricas y aprenda a lidiar con el error humano.

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: Fracción porcentual de ruido.
    - noise_bound: Límite duro en píxeles.

    Returns:
    - box_coords: Coordenadas sucias de la caja.
    - box_labels: Etiquetas identificatorias (Top-Left es 2, Bottom-Right es 3).
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Toma la máscara real (Ground Truth) y la máscara que adivinó el modelo. 
    Calcula dónde se equivocó el modelo (los errores) y simula que un usuario 
    hace un clic aleatorio exactamente en esa zona de error para corregirlo.

    Inputs:
    - gt_masks: Máscaras perfectas reales [B, 1, H_im, W_im].
    - pred_masks: Máscaras predecidas por la red [B, 1, H_im, W_im].
    - num_pt: Cantidad de clics a simular.

    Outputs:
    - points: Las (x, y) de los clics simulados.
    - labels: Si el clic cayó en un Falso Negativo, devuelve 1 (Clic Positivo, añade esto). 
      Si cayó en un Falso Positivo, devuelve 0 (Clic Negativo, borra esto).
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Método de muestreo Inteligente/RITM. 
    A diferencia de la función anterior (que elige un píxel de error al azar), 
    este algoritmo usa OpenCV para medir "distancias", encontrando matemáticamente 
    el punto MÁS PROFUNDO (el epicentro o "núcleo") de la mancha de error, 
    y simula que el usuario hace el clic corrector ahí.
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        # Transformada de Distancia: A cada píxel blanco le asigna un número según
        # qué tan lejos esté del borde más cercano. El valor máximo es el centro geométrico.
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    """
    Función envoltorio (wrapper). Dependiendo del `method` configurado, utiliza 
    la heurística aleatoria ('uniform') o la geométrica ('center') para simular 
    el siguiente clic humano durante el entrenamiento interactivo.
    """
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")