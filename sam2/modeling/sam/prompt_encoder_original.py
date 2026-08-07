# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
CODIFICADOR DE INDICACIONES (Prompt Encoder) - SAM 2
=============================================================================
Este módulo es la "interfaz de usuario" de la red neuronal. 
Toma las interacciones humanas (clics y dibujos) y las traduce a vectores 
matemáticos que el Decodificador puede entender para saber *qué* tiene que segmentar.

CONCEPTOS CLAVES:
- Sparse Prompts (Indicaciones Dispersas): Información que ocupa poco espacio, 
  como Puntos (X, Y) o Cajas (X_min, Y_min, X_max, Y_max).
- Dense Prompts (Indicaciones Densas): Información que cubre toda la imagen, 
  como una máscara de baja resolución provista por un ciclo anterior.
- Tokenización de la Intención: Para que la red sepa qué significa un punto, 
  le sumamos un "Embedding" entrenable dependiendo de si el usuario quiere incluir 
  ese punto (Positivo), excluirlo (Negativo), o si es la esquina de una caja.
=============================================================================
"""

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Codifica las indicaciones (prompts) para alimentar al decodificador de máscaras.

        Argumentos:
          embed_dim (int): Dimensión del espacio matemático de los prompts (ej. 256).
          image_embedding_size (tuple(int, int)): El tamaño (Alto, Ancho) al que se 
            redujo la imagen al pasar por el Backbone Hiera (ej. 64x64).
          input_image_size (int): El tamaño original de la imagen introducida 
            a la red (ej. 1024x1024).
          mask_in_chans (int): La cantidad de canales ocultos usados al comprimir 
            una máscara dibujada (Dense Prompt).
          activation (nn.Module): La función de activación no lineal usada.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        
        # Genera coordenadas matemáticas aleatorias (Fourier Features) para los clics.
        # Usa la mitad de la dimensión porque se codificará X e Y de forma separada.
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # TOKENS DE ESTADO PARA LOS PUNTOS:
        # 0: Clic Negativo (Fondo, no quiero esto)
        # 1: Clic Positivo (Frente, quiero esto)
        # 2: Esquina superior izquierda de una caja delimitadora
        # 3: Esquina inferior derecha de una caja delimitadora
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        
        # Un token "fantasma" para rellenar matrices cuando no hay puntos (Padding).
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        
        # Red convolucional para reducir las máscaras introducidas por el usuario
        # (o por un fotograma previo) para que coincidan con la resolución del Image Embedding.
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        # Token especial que se usa si el usuario NO proporcionó ninguna máscara previa.
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Devuelve la grilla matemática posicional completa de la imagen.
        El Mask Decoder necesita esto para entender la distribución espacial 
        general antes de empezar a dibujar.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """
        Convierte los clics (X, Y) y sus etiquetas (Positivo/Negativo) en tensores profundos.
        """
        # Desplaza matemáticamente el clic hacia el centro exacto del píxel 
        # (ej. de la coordenada 1.0 a la coordenada 1.5) para evitar errores de redondeo.
        points = points + 0.5  # Shift to center of pixel
        
        # Si no hay cajas (solo clics), SAM requiere que SIEMPRE exista al menos un 
        # punto "fantasma" de relleno al final de la secuencia para funcionar correctamente.
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            
        # Asigna la coordenada matemática Fourier al punto físico
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        # MAGIA SEMÁNTICA: Suma el "significado" (Token) a la "ubicación" (Embedding).
        # Si el label es -1 (Fantasma), suma el token de no-punto.
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        # Si el label es 0 (Negativo), suma el token de "no quiero este objeto".
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        # Si el label es 1 (Positivo), suma el token de "sí quiero este objeto".
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        # Si el label es 2 (Caja - Arriba Izquierda)
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        # Si el label es 3 (Caja - Abajo Derecha)
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convierte las Cajas Delimitadoras (Bounding Boxes) en tensores.
        Técnicamente, SAM trata una caja como si fueran 2 clics especiales.
        """
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        
        # Suma el token de "Esquina Superior Izquierda" a la primera coordenada.
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        
        # Suma el token de "Esquina Inferior Derecha" a la segunda coordenada.
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Convierte una máscara en un tensor denso usando convoluciones."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Infiere inteligentemente el tamaño del lote (batch size) basándose en cuál 
        de los tres tipos de entrada está presente.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        """Utilidad para saber en qué hardware (GPU/CPU) están cargados los pesos."""
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        El método orquestador de este módulo. 
        Recibe todas las interacciones del usuario y las empaqueta en dos grandes tensores: 
        uno disperso (para puntos/cajas) y uno denso (para máscaras completas).

        Argumentos:
          points: Tupla de (Coordenadas [X, Y], Etiquetas [0 o 1]).
          boxes: Cajas en formato XYXY.
          masks: Matriz binaria 2D del mismo tamaño que la imagen original.

        Retorna:
          torch.Tensor: Embeddings dispersos (Sparse). Un vector concatenado con la 
            información matemática de todos los puntos y cajas introducidos.
          torch.Tensor: Embeddings densos (Dense). Una matriz matemática tridimensional 
            (C, H, W) con la información de las máscaras previas.
        """
        bs = self._get_batch_size(points, boxes, masks)
        
        # Inicializa un tensor vacío en la GPU para ir apilando (concatenando) los puntos
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        
        if points is not None:
            coords, labels = points
            # Llama al sub-método, aplicando el Padding Fantasma solo si no hay Cajas presentes.
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            # Si el usuario solo hizo clics y no pasó ninguna máscara previa, 
            # crea una grilla "en blanco" llena con el token especial de `no_mask_embed`,
            # expandiéndola para que ocupe todo el alto y ancho requerido.
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings