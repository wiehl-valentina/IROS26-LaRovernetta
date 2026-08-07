# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
CODIFICADOR DE MEMORIA (Memory Encoder) - SAM 2
=============================================================================
Este módulo es el responsable de crear los "recuerdos" del robot a medida que 
procesa un video. Cuando el modelo predice la forma de un obstáculo en el 
frame actual, este código toma esa máscara de alta resolución, la comprime, 
la mezcla con la imagen original y la empaqueta para guardarla en el Banco de 
Memoria, permitiendo que el robot no "olvide" lo que acaba de ver.
=============================================================================
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d


class MaskDownSampler(nn.Module):
    """
    CONCEPTOS CLAVES:
    - Downsampling (Submuestreo): Toma una máscara de alta resolución (ej. 256x256) 
      y la reduce geométricamente (ej. a 64x64) para ahorrar memoria RAM y tiempo 
      de cálculo en los frames futuros.
    - Trade-off (Resolución vs Canales): A medida que la imagen se hace más pequeña 
      en alto y ancho (stride**2), se compensa aumentando la cantidad de "canales" 
      o "filtros" de profundidad. Es decir, pierde resolución espacial pero gana 
      profundidad conceptual.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                # Convolución para reducir el tamaño de la matriz
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            # Normalización para estabilizar el entrenamiento
            self.encoder.append(LayerNorm2d(mask_out_chans))
            # Activación no lineal GELU (Gaussian Error Linear Unit)
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        # Proyección final para alinear exactamente el número de canales con 
        # las características visuales (embed_dim = 256).
        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""
    Bloque fundacional inspirado en ConvNeXt, una arquitectura de visión 
    por computadora de vanguardia.
    
    CONCEPTOS CLAVES:
    - Depthwise Convolution (DwConv): En lugar de hacer una convolución tradicional 
      pesada donde todos los canales de color/profundidad interactúan entre sí, 
      DwConv aplica un filtro separado a cada canal de forma independiente. Esto 
      es computacionalmente baratísimo y extremadamente rápido.
    - Pointwise Convolution (1x1 Conv): Después del DwConv, se usa un filtro 1x1 
      (lineal) para volver a mezclar la información entre canales.
    - DropPath: Apaga completamente el bloque de forma aleatoria durante el 
      entrenamiento. Fuerza a la red a ser redundante y robusta (regularización).
    
    Hay dos implementaciones equivalentes:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; todo en (N, C, H, W)
    (2) DwConv -> Permuta a (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permuta atrás.
    Se utiliza (2) ya que se descubrió que es ligeramente más rápido en PyTorch.
    
    Args:
        dim (int): Número de canales de entrada.
        drop_path (float): Tasa de profundidad estocástica. Por defecto: 0.0
        layer_scale_init_value (float): Valor inicial para Layer Scale. Por defecto: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    """
    Fusionador de Características (Fuser).
    Toma la suma cruda de las características de la imagen y la máscara submuestreada, 
    y las pasa por múltiples bloques convolucionales (CXBlock) para que la red 
    "digiera" la mezcla y entienda cómo la máscara recubre el objeto visualmente.
    """
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    """
    El orquestador principal del proceso de memorización.
    
    Flujo de trabajo:
    1. Recibe los logits de la máscara generada (blanco/negro).
    2. Convierte esos logits a probabilidades [0, 1] con función Sigmoide.
    3. Reduce el tamaño de la máscara (MaskDownSampler).
    4. Proyecta las características visuales (pix_feat_proj) para alinear canales.
    5. Suma ambas matrices: Imagen + Máscara.
    6. Las fusiona profundamente (Fuser).
    7. Agrega Codificación Posicional para que la memoria sepa dónde estaba el 
       objeto espacialmente en la pantalla.
    8. Lo devuelve como un "Recuerdo" (Memory Embedding) listo para ser 
       guardado en el historial.
    """
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        
        # SUMA CRÍTICA: Aquí es donde se fusiona lo visual (x) con la intención geométrica (masks)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        # Inyectar noción espacial para que el recuerdo sepa sus coordenadas absolutas
        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}