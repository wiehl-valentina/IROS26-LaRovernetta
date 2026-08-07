# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
ATENCIÓN DE MEMORIA (Memory Attention) - SAM 2
=============================================================================
Este módulo define la arquitectura del Transformer que procesa la información
temporal en el modelo SAM 2. Actúa como el "puente" entre el presente y el pasado.

Su función principal es tomar las características visuales del fotograma actual
(curr/tgt) y compararlas (atenderlas) con las características guardadas en la
memoria (memory) de los fotogramas anteriores, permitiendo al modelo rastrear 
objetos en movimiento, manejar oclusiones y mantener coherencia temporal.
=============================================================================
"""

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):
    """
    Una capa individual del bloque de Atención de Memoria (Transformer Block).
    
    CONCEPTOS CLAVES DE LA ARQUITECTURA TRANSFORMER:
    Esta capa sigue la estructura clásica de un Transformer Decoder, pero adaptada
    para procesar memorias espaciales-temporales:
    
    1. Self-Attention (Atención Propia): El fotograma actual se mira a sí mismo. 
       Busca patrones y relaciones internas (ej: "este píxel del pelaje del perro 
       está relacionado con este otro píxel de la oreja").
    2. Cross-Attention (Atención Cruzada): El fotograma actual (Queries) "pregunta" 
       a las memorias pasadas (Keys/Values). (ej: "Basado en la forma del perro 
       ahora, ¿dónde estaba en los frames anteriores?").
    3. MLP (Multi-Layer Perceptron / Red Neuronal Densa): Procesa y mezcla 
       la información extraída por los pasos anteriores de forma no lineal.
    
    - Positional Encoding (Pos Enc): Inyecta las coordenadas (x,y) a los vectores. 
      Sin esto, la red vería la imagen como una "sopa de píxeles" sin orden espacial.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        # Red neuronal clásica de 2 capas que expande la dimensión (para abstraer conceptos)
        # y la vuelve a comprimir (d_model -> dim_feedforward -> d_model).
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Capas de Normalización (LayerNorm): Estabilizan el entrenamiento evitando 
        # que los números crezcan demasiado al pasar por las multiplicaciones matriciales.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout: Apaga aleatoriamente neuronas para evitar sobreajuste (overfitting).
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        # Define en qué etapas matemáticas exactas se suma la información espacial
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        """
        Ejecuta el bloque de Self-Attention.
        tgt (target): Las características visuales del frame actual.
        query_pos: El mapa de coordenadas espaciales del frame actual.
        """
        # Pre-Norm (normalizar antes de atender estabiliza Transformers profundos)
        tgt2 = self.norm1(tgt)
        
        # En Self-Attention, la Consulta (Query/q) y la Clave (Key/k) son lo mismo.
        # Si está configurado, sumamos la codificación posicional a Q y K.
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        
        # V (Value) se pasa puro sin codificación posicional.
        tgt2 = self.self_attn(q, k, v=tgt2)
        
        # Conexión Residual (Residual Connection): Sumamos el resultado (tgt2) 
        # al input original (tgt). Esto permite que el gradiente fluya directo.
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        """
        Ejecuta el bloque de Cross-Attention.
        Aquí es donde el presente (tgt) se cruza con el pasado (memory).
        
        CONCEPTOS CLAVES:
        - RoPE (Rotary Position Embedding): Es un tipo especial de codificación posicional 
          que usa multiplicaciones complejas en lugar de sumas para enseñar posiciones 
          *relativas* (ej: "A está a 5 píxeles a la derecha de B") en lugar de absolutas.
        - num_k_exclude_rope: Permite excluir ciertos tokens (como el Object Pointer, que 
          es abstracto y no tiene una posición física (X,Y)) del cálculo de RoPE.
        """
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            # Query: El fotograma actual
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            # Key: Las memorias de los fotogramas pasados
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            # Value: El "contenido" puro de las memorias pasadas
            v=memory,
            **kwds,
        )
        # Conexión Residual
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """
        Pase hacia adelante (Forward Pass) de la capa completa.
        Pasa los datos en cascada: Self-Attention -> Cross-Attention -> Red Neuronal (MLP).
        """

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        
        # MLP (Perceptrón Multicapa / Red Neuronal Feedforward)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    """
    El módulo contenedor principal de la Atención de Memoria.
    Apila secuencialmente múltiples `MemoryAttentionLayer` (ej. 4 capas).
    
    CONCEPTOS CLAVES:
    - Profundidad (num_layers): Pasar el tensor de memoria a través de 4 capas permite 
      al modelo realizar un razonamiento progresivamente más complejo (ej. Capa 1: "se movió 
      a la derecha", Capa 4: "el objeto se ocultó parcialmente detrás del poste").
    - Transposición de Batch (batch_first): En PyTorch, los Transformers clásicos 
      esperan los tensores con la forma [Secuencia, Lote, Canales] (Seq-first) por eficiencia 
      histórica. Esta clase convierte automáticamente desde y hacia [Lote, Secuencia, Canales] 
      (Batch-first), que es la forma estándar moderna.
    """
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        
        # Clonar la arquitectura de la capa `num_layers` veces de manera independiente
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs (fotograma actual)
        memory: torch.Tensor,  # cross-attention inputs (historial/banco de memoria)
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        # Acondicionamiento inicial si se recibe una lista envuelta de longitud 1
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        # Inyección posicional global temprana, atenuada por un factor de 0.1
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            # Transpone (cambia el orden de las dimensiones) de (Batch, Seq, Features) 
            # a (Seq, Batch, Features) para alimentar la matemática del Transformer subyacente.
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        # Bucle principal: El output de una capa alimenta como input a la siguiente
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
            
        # Normalización final post-procesamiento de todas las capas
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            # Des-transpone para devolver al formato amigable BxSxC
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output