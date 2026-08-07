# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
CODIFICADORES POSICIONALES (Positional Embeddings) - SAM 2
=============================================================================
Este módulo define las técnicas matemáticas que le enseñan a las redes neuronales 
(específicamente a los Transformers) sobre el "espacio" y la "geometría". 

CONCEPTOS CLAVES:
- Problema del Transformer: Un Transformer ve una imagen como una lista desordenada 
  de píxeles. No sabe qué píxel está arriba, abajo, a la izquierda o a la derecha.
- Solución (Positional Encoding): Se le inyecta una "coordenada matemática" a cada píxel.
  Existen varias estrategias aquí implementadas:
  1. Sine/Cosine (Absoluta): Usa ondas trigonométricas, popularizada por "Attention Is All You Need".
  2. Random/Gaussian (Frecuencias): Usada para los Puntos y clics del usuario.
  3. RoPE (Rotatoria): Transforma las coordenadas usando números complejos, permitiendo 
     al modelo entender distancias *relativas* de forma nativa.
=============================================================================
"""

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    Codificación posicional absoluta usando ondas Senoidales y Cosinusoidales.
    
    Es una versión generalizada a 2D (imágenes) del embedding clásico de los Transformers.
    Por cada coordenada (X, Y) de la imagen, genera un vector donde la mitad de los 
    valores corresponden a funciones seno/coseno del eje X, y la otra mitad del eje Y, 
    usando múltiples frecuencias (desde ondas muy largas hasta ondas muy cortas).
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        # Following settings only relevant
        # for warmping up cache for compilation
        warmup_cache: bool = True,
        image_size: int = 1024,
        strides: Tuple[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        # Divide la capacidad a la mitad: una mitad para X, la otra para Y
        self.num_pos_feats = num_pos_feats // 2
        # Frecuencia base para generar las ondas trigonométricas
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        # Warmup: Pre-calcula y guarda en la RAM de la GPU las matrices posicionales 
        # de los tamaños más comunes para no tener que regenerarlas en cada iteración.
        if warmup_cache and torch.cuda.is_available():
            # Warmup cache for cuda, to help with compilation
            device = torch.device("cuda")
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, device, *cache_key)

    def _encode_xy(self, x, y):
        """
        Toma coordenadas crudas X e Y y les aplica la expansión trigonométrica 
        para generar un espacio latente de alta dimensionalidad.
        """
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        # Intercala senos y cosenos para generar el vector final
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        """
        Codifica espacialmente una Caja Delimitadora (Bounding Box).
        Concatena las ondas del centro (X, Y) con los valores puros de Ancho y Alto.
        """
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """
        Codifica puntos individuales (los clics del usuario).
        A diferencia de los píxeles de una imagen, los puntos llevan pegada una 
        etiqueta (label) que indica si es un clic positivo (quiero esto) o 
        negativo (no quiero esto).
        """
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        """
        Genera la matriz posicional 2D completa (para todos los píxeles de una imagen).
        Si el tamaño ya fue calculado antes, lo devuelve instantáneamente desde el caché.
        """
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        # Genera mallas 2D con los índices de fila (y) y columna (x)
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        # Concatena el eje Y y el eje X y ajusta las dimensiones a CxHxW
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, *cache_key)


class PositionEmbeddingRandom(nn.Module):
    """
    Codificación posicional usando frecuencias espaciales aleatorias.
    
    A diferencia de la clase anterior que usa frecuencias predecibles (seno de 1, 
    seno de 2...), esta clase multiplica las coordenadas por una matriz Gaussiana 
    generada al azar (Fourier Features). Está demostrado matemáticamente que esto 
    ayuda a las redes neuronales a capturar detalles finos (altas frecuencias) 
    mucho mejor en problemas de segmentación visual.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # Guarda la matriz aleatoria en el modelo para que no cambie entre inferencias
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# """
# =============================================================================
# RoPE (Rotary Positional Encoding)
# =============================================================================
# Una de las innovaciones más fuertes de SAM 2 y de Modelos de Lenguaje (LLMs) como LLaMA.
# En lugar de SUMAR un vector de posición a los datos visuales, RoPE convierte 
# los vectores a un plano de números complejos y los ROTA un cierto ángulo.
# 
# ¿Por qué rotar?
# Matemáticamente, la diferencia de ángulo entre dos vectores rotados depende ÚNICAMENTE 
# de la distancia que hay entre ellos, sin importar en qué parte exacta de la imagen estén.
# Esto otorga "Invarianza Traslacional": el modelo entiende que "el ojo está al lado 
# de la nariz" sin importar si la cara está en el centro de la foto o en un rincón.
# =============================================================================
# """

# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    """
    Inicializa una malla de posiciones espaciales absolutas (0 a ancho, 0 a alto).
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    Pre-calcula la tabla de frecuencias de rotación complejas (cis = cos + i*sin).
    Genera los ángulos exactos con los que los tensores deberán ser rotados 
    basándose en su coordenada X y su coordenada Y.
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    # torch.polar genera un número complejo a partir de una magnitud (1.0) y un ángulo (freqs)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Ajusta las dimensiones de la tabla de frecuencias para que pueda multiplicarse 
    (mediante broadcasting de PyTorch) contra tensores masivos que vienen en lotes (batch).
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    Aplica físicamente la rotación a las matrices de Consulta (Query/xq) y 
    Clave (Key/xk) dentro del mecanismo de Atención.
    
    CONCEPTOS CLAVES:
    - Matemáticas Complejas: Convierte los tensores flotantes crudos en números 
      complejos (`torch.view_as_complex`), los multiplica por la tabla de ángulos 
      (lo cual en el plano complejo equivale a rotarlos), y luego los devuelve 
      como números reales (`torch.view_as_real`).
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Esta es la línea mágica donde ocurre la rotación real: xq_ * freqs_cis
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
            
    # Rota también la clave (Key)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)