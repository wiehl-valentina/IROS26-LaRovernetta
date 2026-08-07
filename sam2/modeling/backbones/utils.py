# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
UTILIDADES PARA EL BACKBONE (sam2.modeling.backbones.utils) - SAM 2
=============================================================================
Este módulo proporciona herramientas matemáticas fundamentales para el 
procesamiento de imágenes dentro de la arquitectura de Transformers de Visión (ViT).

CONCEPTOS CLAVES:
1. "Patch Embedding" (Tokenización de la Imagen): Los Transformers originales 
   fueron diseñados para leer secuencias de palabras (texto). Para que puedan 
   leer una imagen, primero debemos "picarla" en pequeños recuadros (parches) 
   y convertir cada recuadro en un vector matemático (Token/Embedding).
2. "Window Partitioning" (Partición por Ventanas): La operación matemática de 
   Atención (Self-Attention) tiene una complejidad cuadrática $O(N^2)$. Si 
   intentamos que cada píxel atienda a todos los demás píxeles en una imagen 
   de alta resolución, la memoria RAM de la GPU explotaría. La solución es 
   dividir la imagen en "ventanas" (cuadrículas aisladas) y calcular la 
   atención únicamente entre los píxeles que están dentro de la misma ventana.
=============================================================================
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
---------------------------------------------------------------------------
1. PARTICIÓN POR VENTANAS (Window Partition)
---------------------------------------------------------------------------
Toma un mapa de características visuales gigante y lo trocea en ventanas 
perfectamente cuadradas y no superpuestas (non-overlapping).
---------------------------------------------------------------------------
"""
def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
        
    CONCEPTOS CLAVES INTERNOS:
    - Padding (Relleno): ¿Qué pasa si la imagen tiene un ancho de 100 píxeles 
      pero queremos dividirla en ventanas de 14x14? 100 no es divisible por 14 
      (sobran 2 píxeles). Para no perder información ni romper la matriz, la 
      función agrega píxeles vacíos artificiales (padding) en los bordes para 
      alcanzar el múltiplo más cercano (en este caso, rellenaría hasta 112).
    - Remodelado (Reshape & Permute): Usa manipulación tensorial avanzada 
      para reordenar las dimensiones de [Lote, Alto, Ancho, Canales] a 
      [Lote * Cantidad_de_Ventanas, Tamaño_Ventana, Tamaño_Ventana, Canales].
    """
    B, H, W, C = x.shape

    # Calcula cuánto relleno (padding) se necesita para ser múltiplo exacto del window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        # Aplica el relleno en la parte inferior (pad_h) y derecha (pad_w) del tensor
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # Reorganiza la matriz para aislar las ventanas espacialmente
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # Colapsa las dimensiones para que el batch contenga múltiples ventanas independientes
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


"""
---------------------------------------------------------------------------
2. DES-PARTICIÓN POR VENTANAS (Window Unpartition)
---------------------------------------------------------------------------
Realiza exactamente el proceso inverso a la función anterior.
Una vez que el Transformer terminó de procesar la atención dentro de las 
pequeñas ventanas, necesitamos volver a pegarlas todas juntas como si 
fueran un rompecabezas para reconstruir la imagen completa.
---------------------------------------------------------------------------
"""
def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
        
    CONCEPTOS CLAVES INTERNOS:
    - Un-padding (Recorte): Como en el paso anterior se agregaron píxeles artificiales 
      para que la división fuera matemáticamente perfecta, aquí se extirpan esos 
      píxeles extra (`x[:, :H, :W, :]`) para devolver exactamente el tensor del 
      tamaño original.
    """
    Hp, Wp = pad_hw
    H, W = hw
    # Calcula el tamaño del Lote (Batch) original deduciéndolo de la cantidad total de ventanas
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    
    # Desenrolla las ventanas
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    # Vuelve a poner el Alto y el Ancho en sus dimensiones correctas
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    # Elimina (recorta) los píxeles de relleno (padding) si es que se agregaron
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


"""
---------------------------------------------------------------------------
3. EMBEDDING DE PARCHES (PatchEmbed)
---------------------------------------------------------------------------
Esta es la "Puerta de Entrada" física de la imagen a la red neuronal.
Es la responsable de traducir los colores RGB de una foto en un lenguaje 
de alta dimensionalidad que el Transformer pueda razonar.
---------------------------------------------------------------------------
"""
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    CONCEPTOS CLAVES:
    - ¿Cómo tokenizar una foto? En lugar de hacer operaciones complejas, la 
      comunidad científica descubrió que una simple Convolución 2D (`nn.Conv2d`) 
      con un tamaño de "paso" (stride) igual al tamaño del parche es la forma 
      más eficiente de hacerlo.
    - Ejemplo: Si el `stride` es (4, 4) y el `kernel` es (7, 7), la red lanza un 
      cuadrado recolector sobre la imagen que se va moviendo de 4 en 4 píxeles, 
      absorbiendo la información de color y comprimiéndola en un vector denso 
      de tamaño `embed_dim` (ej. 96 o 768 canales).
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,  # Normalmente 3 canales para imágenes RGB
        embed_dim: int = 768,  # El tamaño del "vocabulario" matemático interno de la red
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        # La proyección es literalmente una capa convolucional que reduce la 
        # resolución espacial pero aumenta drásticamente la profundidad semántica.
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pasa la imagen RGB por la convolución
        x = self.proj(x)
        
        # B C H W -> B H W C
        # Permutación vital: PyTorch procesa imágenes con los Canales al principio (C, H, W).
        # Sin embargo, la matemática estándar de los Transformers requiere que la 
        # dimensión de características (los canales) esté al final del tensor (H, W, C).
        x = x.permute(0, 2, 3, 1)
        return x