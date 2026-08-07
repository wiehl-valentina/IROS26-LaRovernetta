# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
BACKBONE "HIERA" (Columna Vertebral de Visión) - SAM 2
=============================================================================
Este archivo define la red neuronal encargada de la "Percepción Visual Cruda".
Hiera (Hierarchical Vision Transformer) es una evolución altamente optimizada 
de los Vision Transformers (ViT). 

CONCEPTOS CLAVES:
1. Arquitectura Jerárquica (Hierarchical): A diferencia de un ViT clásico que 
   mantiene la misma resolución en toda la red, Hiera imita a las redes 
   convolucionales. Empieza viendo la imagen en alta resolución (para captar 
   bordes y texturas) y, a medida que profundiza, va "achicando" la imagen 
   (Pooling) pero "ensanchando" los canales para captar conceptos abstractos 
   (ej. "esto es una rueda").
2. Atención por Ventanas (Window Attention): Para no explotar la memoria RAM 
   comparando cada píxel con absolutamente todos los demás píxeles de una imagen 
   gigante, Hiera divide la imagen en pequeñas "ventanas" (grids) y hace atención 
   solo dentro de esas ventanas. Ocasionalmente, usa "Atención Global" para 
   que las ventanas se comuniquen entre sí.
=============================================================================
"""

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """
    Función auxiliar para aplicar Pooling (reducción de tamaño, usualmente MaxPool) 
    sobre un tensor que viene con formato de Transformer.

    El Transformer usa el formato (Batch, Alto, Ancho, Canales) o (B, H, W, C).
    Sin embargo, las funciones de Pooling de PyTorch (optimizadas para imágenes) 
    exigen el formato de Convolución: (Batch, Canales, Alto, Ancho) o (B, C, H, W).
    Esta función hace la permutación de ida y vuelta.
    """
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """
    Atención Multi-Escala (MultiScale Attention).
    
    Es una variación del mecanismo de Atención tradicional. Su particularidad 
    es que permite introducir un `q_pool` (Pooling de Queries). 
    
    CONCEPTOS CLAVES:
    - Transición de Etapas: Cuando la red necesita pasar de una etapa de alta 
      resolución a una de baja resolución (Stage Shift), aplica un filtro (Pool) 
      a las 'Preguntas' (Queries / q). Como resultado, la matriz de salida de la 
      Atención será físicamente más pequeña (en Alto y Ancho) que la matriz de entrada.
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        # Proyección lineal única que escupe Q, K y V al mismo tiempo (dim_out * 3)
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # Desempaqueta el tensor gigante en 3 tensores: Queries (q), Keys (k), Values (v)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        # Si estamos en un límite de etapa, achica el tensor de Consultas (Q)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        # Utiliza la Atención Flash (Scaled Dot Product Attention) nativa de PyTorch 
        # que está escrita en C++ y CUDA para ser exponencialmente más rápida.
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """
    Bloque Fundamental del Transformer Hiera.
    
    Apila secuencialmente:
    1. Normalización.
    2. Partición por Ventanas (Divide la imagen en grillas pequeñas para ahorrar RAM).
    3. MultiScale Attention (Dentro de las ventanas, o Global si está configurado).
    4. Des-partición de Ventanas (Vuelve a armar la imagen).
    5. MLP (Red Densa para procesar los hallazgos de la atención).
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        # Si la dimensión de salida es diferente (porque cruzamos una Etapa), 
        # proyecta el "Atajo" (Shortcut) para que los tensores se puedan sumar después.
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        # Trocea la imagen grande en pequeños cuadrados (Ventanas) de tamaño `window_size`
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            # Si hubo reducción espacial, recalcula los tamaños de los parches
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        # Vuelve a pegar todas las ventanas (ya procesadas) para formar la imagen completa
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        # Conexión Residual
        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    
    El orquestador maestro del Backbone Visual.
    Instancia y apila jerárquicamente múltiples bloques `MultiScaleBlock`.
    
    El parámetro vital es `stages` (ej. [2, 3, 16, 3]). Esto significa que agrupa 
    el procesamiento en 4 fases. Después de la fase 1 (2 bloques), reduce la resolución a la mitad 
    y duplica los canales. Al terminar la fase 2 (3 bloques), lo vuelve a hacer, etc.
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        # Índices exactos donde termina una etapa y empieza otra (Stage Shift)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        # Patch Embed: La puerta de entrada. Toma los píxeles RGB y los agrupa en 
        # "parches" (tokens) para que el Transformer empiece a digerirlos.
        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        # Estrategia especial para indicar posiciones X,Y. Combina una posición global (bkg)
        # y una posición local detallada para los elementos de cada ventana.
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        # CONSTRUCCIÓN DE LA RED
        # Crea bloque por bloque ajustando canales y resolución según la etapa correspondiente.
        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                # Si este índice es global, apaga el Window Size (size=0)
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                # Transición de Etapa: Duplica los canales (dim_out) y las Cabezas de Atención
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        # Guarda qué tamaño de canales tiene el final de cada etapa 
        # (vital para el Feature Pyramid Network (FPN) que las mezclará luego).
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """
        Calcula e inyecta la información espacial para que la red sepa 
        dónde está parado cada token en la imagen. Fusiona una grilla global 
        interpolada con la grilla repetitiva de las ventanas.
        """
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Flujo de datos de la imagen cruda a través de la columna vertebral.
        Recibe píxeles y devuelve un arreglo (List) con los mapas de características 
        extraídos al final de CADA etapa (de más borroso/general a más nítido/específico).
        """
        # Convierte los píxeles a tokens matemáticos
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        # Pasa los tokens por todos los bloques en cascada
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # Si estamos en el último bloque de una etapa, guarda el resultado
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        """
        Función auxiliar utilizada principalmente por Optimizadores (como AdamW) 
        para aplicar algoritmos avanzados como "Layer-wise Learning Rate Decay" 
        (Cambiar la tasa de aprendizaje dependiendo de la profundidad de la capa).
        """
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        """Devuelve la cantidad total de bloques de procesamiento en la red."""
        return len(self.blocks)