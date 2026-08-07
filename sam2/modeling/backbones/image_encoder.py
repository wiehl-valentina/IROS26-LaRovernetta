# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
CODIFICADOR DE IMAGEN (Image Encoder) Y CUELLO FPN (FpnNeck) - SAM 2
=============================================================================
Este módulo define las estructuras de alto nivel que orquestan el procesamiento 
visual de la imagen de entrada antes de enviarla a los Transformers y al Decodificador.

CONCEPTOS CLAVES:
- Trunk (Tronco / Columna Vertebral): Es la red neuronal extractora base (como Hiera). 
  Su trabajo es "mirar" la imagen y generar múltiples mapas de características, 
  desde los más grandes (detalles finos como bordes) hasta los más pequeños 
  (conceptos semánticos como "es un coche").
- Neck (Cuello): Es el puente entre el Trunk y las cabezas de predicción. 
  Aquí se usa un FPN (Feature Pyramid Network) para combinar y estandarizar 
  esos mapas de características.
- Scalp (Recortar / "Escalpar"): Operación para descartar opcionalmente el mapa 
  de características más pequeño/profundo si no se necesita para la arquitectura actual.
=============================================================================
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Contenedor principal (Wrapper) del codificador de imágenes.
    Encapsula el 'Trunk' (Hiera) y el 'Neck' (FPN) en un solo módulo cohesivo.
    """
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        # Validación de seguridad: Asegura que la cantidad de canales que escupe 
        # el tronco en cada etapa coincida exactamente con la cantidad de canales 
        # que espera recibir el cuello.
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        """
        Pase hacia adelante (Forward Pass).
        Toma el tensor de la imagen (sample), lo pasa por el tronco para obtener 
        la pirámide visual, y luego por el cuello para fusionar los niveles.
        """
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            # Elimina las N características más profundas de la pirámide si scalp > 0
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        # La característica fuente principal suele ser la de mayor resolución 
        # que sobrevive al recorte.
        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    
    CONCEPTOS CLAVES DEL FPN (Red de Pirámide de Características):
    El FPN resuelve un problema clásico en visión por computadora. Las capas iniciales 
    del Trunk ven la imagen en alta resolución pero no entienden "qué" están viendo 
    (poca semántica). Las capas profundas entienden perfectamente "qué" es el objeto, 
    pero en tan baja resolución que no saben exactamente "dónde" están sus bordes.
    
    El FPN soluciona esto con una ruta "Top-Down" (De Arriba hacia Abajo): Toma el 
    mapa profundo (pequeño y muy inteligente), lo estira (interpolación), y lo suma 
    al mapa anterior (más grande pero menos inteligente). Al hacer esto en cascada, 
    todos los mapas terminan siendo altamente inteligentes y con excelente resolución espacial.
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        
        Inicializa las proyecciones lineales (Conv2d 1x1) que igualan la dimensión 
        de todos los canales a `d_model` (ej. 256) para que luego puedan sumarse matemáticamente.
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        
        # Crea una capa convolucional por cada nivel de la pirámide para unificar canales
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        """
        Ejecuta la lógica Top-Down del FPN.
        Recibe la lista de características multiescala `xs` extraídas por el Trunk.
        """

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        # Itera en reversa: Comienza desde el mapa más pequeño/profundo y va subiendo 
        # hacia el más grande/superficial.
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            # Paso Lateral (Lateral Connection): Unifica la cantidad de canales a d_model
            lateral_features = self.convs[n - i](x)
            
            # Paso Top-Down: Si el nivel actual tiene habilitada la propagación y hay 
            # una característica previa (de menor resolución), agranda la previa y la suma.
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,  # Agranda el tamaño espacial (H, W) al doble
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                # Fusión: Combina la información local (lateral) con el contexto global (top-down)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
                
            x_out = prev_features
            out[i] = x_out
            # Genera dinámicamente la codificación posicional para este nivel de la pirámide
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos