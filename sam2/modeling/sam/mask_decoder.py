# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================================
DECODIFICADOR DE MÁSCARAS (Mask Decoder) - SAM 2
=============================================================================
Este es el "Lápiz" del modelo. Una vez que la imagen ha sido codificada y 
fusionada con los clics del usuario y la memoria del pasado, este módulo toma 
toda esa abstracción matemática y literalmente "dibuja" la máscara final píxel 
por píxel.

CONCEPTOS CLAVES:
- TwoWayTransformer: El núcleo del decoder. Permite una atención bidireccional 
  donde los clics del usuario actualizan la imagen, y la imagen actualiza a su 
  vez el significado de los clics.
- Hiperredes (Hypernetworks): En lugar de tener convoluciones fijas, este modelo 
  genera dinámicamente los pesos de la convolución basándose en el "Token de 
  Máscara". Es decir, el modelo crea un filtro específico al vuelo solo para 
  pintar el objeto que el usuario seleccionó.
=============================================================================
"""

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predice las máscaras dado el embedding de la imagen y los embeddings 
        de los prompts (clics), utilizando una arquitectura Transformer.

        Argumentos:
          transformer_dim (int): La dimensión de los canales del transformer.
          transformer (nn.Module): El TwoWayTransformer usado para mezclar datos.
          num_multimask_outputs (int): Cantidad de opciones que el modelo da cuando 
            un clic es ambiguo (por defecto 3: Ej. Neumático, Rueda completa, Auto entero).
          activation (nn.Module): El tipo de activación no lineal usada al agrandar 
            (upscale) la resolución de la máscara.
          iou_head_depth (int): Profundidad de la red que adivina qué tan buena es la máscara.
          iou_head_hidden_dim (int): Dimensión oculta de esa misma red (MLP).
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # TOKENS ESPECIALES: 
        # Como los transformers son "ciegos" a lo que están procesando, inyectamos estos 
        # tensores entrenables. Actúan como "preguntas" a la red.
        # iou_token: "Oye red, ¿qué tan perfecta te quedó la máscara?"
        self.iou_token = nn.Embedding(1, transformer_dim)
        
        self.num_mask_tokens = num_multimask_outputs + 1
        # mask_tokens: "Oye red, devuélveme 4 vectores que resuman la forma del objeto"
        # (1 vector para la respuesta unívoca segura + 3 vectores para respuestas ambiguas multimáscara).
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            # obj_score_token: "Oye red, ¿el objeto sigue en la foto o desapareció?"
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        # Upsampling (Agrandado): Toma la salida matemática comprimida (ej. 64x64) 
        # y la estira usando Convoluciones Transpuestas para acercarla al tamaño de la foto.
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            # Convoluciones que inyectan los detalles de los bordes perfectos (High Res)
            # extraídos por las capas iniciales del Backbone Hiera.
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        # Hiperredes: Redes neuronales (MLPs) que generan pesos para otras redes.
        # Aquí convierten el resumen abstracto del objeto (Mask Tokens) en filtros 
        # multiplicables que "cortarán" la máscara final del lienzo visual.
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # Cabeza (Head) predictora de IoU. La propia red evalúa su desempeño y 
        # escupe un número del 0.0 al 1.0 indicando su confianza.
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        
        # Cabeza predictora de existencia. Evalúa si el objeto está ocluido/ausente.
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predice las máscaras dado el embedding de imagen y los prompts.

        Argumentos:
          image_embeddings: El tensor visual procesado (con memoria incluida).
          image_pe: Codificación posicional para saber "dónde está arriba y abajo".
          sparse_prompt_embeddings: Clics o Cajas del usuario.
          dense_prompt_embeddings: Máscaras de baja resolución inyectadas por el usuario.
          multimask_output: Si es True, fuerza la salida de 3 opciones (ambigüedad).

        Retorna:
          torch.Tensor: Máscaras generadas.
          torch.Tensor: Puntajes de Confianza (IoU) para esas máscaras.
          torch.Tensor: El "Puntero" abstracto (SAM token) que servirá de memoria.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            # Corta el tensor y devuelve exclusivamente las 3 máscaras ambiguas
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            # Evalúa si la respuesta "única" es sólida. Si el borde es difuso, 
            # cambia inteligentemente la salida a la mejor de las 3 máscaras ambiguas.
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            # Devuelve exclusivamente la máscara N°0 (La respuesta segura y única de SAM).
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """El motor matemático subyacente. Orquesta el TwoWayTransformer y el Upscaling."""
        # Concatenate output tokens
        # Junta todas las "Preguntas" (tokens especiales) con los clics del usuario
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
            
        # Suma literal: La foto base + los garabatos densos que dibujó el usuario
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # FUSIÓN CRUZADA: Los píxeles interrogan a los clics, y los clics a los píxeles.
        hs, src = self.transformer(src, pos_src, tokens)
        
        # Recupera las "Respuestas" del Transformer
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            # U-Net Style Skip Connections: 
            # Inyecta los bordes ultra-nítidos extraídos en los primeros pasos del modelo Hiera 
            # para evitar que la máscara quede "pixelada" al ser agrandada.
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # PRODUCTO PUNTO (Dot Product): La magia final.
        # Pasa los tokens de máscara por las hiperredes, las cuales escupen una 
        # matriz de pesos. Multiplica esa matriz por la imagen upscaleada y ¡PUM!: 
        # recorta matemáticamente la silueta exacta del objeto (crea la máscara).
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # Las otras cabezas escupen la confianza de la IA en lo que acaba de hacer.
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Calcula qué tan 'estable' es una máscara al binarizarse.
        
        CONCEPTOS CLAVES:
        A veces, el modelo dibuja bordes súper borrosos ("Tal vez es auto, tal vez es calle"). 
        Esta función altera ligeramente el umbral de corte matemático (delta). Si la 
        máscara encoge dramáticamente al cambiar el umbral (área u vs área i), significa 
        que el modelo está adivinando y la silueta es "Inestable".
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        Auto-corrección dinámica de la IA sin intervención humana.
        
        Si el usuario pide 1 sola máscara, pero el modelo detecta que esa máscara es 
        muy inestable (ver método anterior), la IA hace trampa a favor del usuario: 
        Muta y elige silenciosamente la mejor de las 3 máscaras ambiguas (que suele 
        tener bordes más definidos) para que el rastreo no colapse en fotogramas futuros.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out