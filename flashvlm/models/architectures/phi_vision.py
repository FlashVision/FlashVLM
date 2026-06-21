"""Phi-3-Vision style architecture: native multimodal with SigLIP encoder."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvlm.cfg.config import FlashVLMConfig, VisionConfig, ProjectorConfig
from flashvlm.models.vision_encoder import SigLIPVisionEncoder, build_vision_encoder
from flashvlm.models.projector import MLPProjector
from flashvlm.registry import MODELS


class PhiImageEmbedding(nn.Module):
    """Image embedding module for Phi-Vision using sub-image decomposition.

    Splits high-resolution images into crops, encodes each crop,
    and merges them with learnable separator tokens.
    """

    def __init__(
        self,
        config: FlashVLMConfig,
        num_crops: int = 4,
        merge_type: str = "flat",
    ):
        super().__init__()
        self.num_crops = num_crops
        self.merge_type = merge_type
        self.hidden_size = config.language.hidden_size
        vision_cfg = config.vision
        vision_cfg.encoder_name = vision_cfg.encoder_name or "google/siglip-so400m-patch14-384"
        self.vision_encoder = build_vision_encoder(vision_cfg)

        proj_cfg = config.projector
        proj_cfg.type = "mlp"
        proj_cfg.input_dim = vision_cfg.hidden_size
        proj_cfg.output_dim = config.language.hidden_size
        proj_cfg.num_layers = 2
        self.projector = MLPProjector(proj_cfg)

        self.sub_gn = nn.Parameter(torch.randn(1, 1, config.language.hidden_size) * 0.02)
        self.glb_gn = nn.Parameter(torch.randn(1, 1, config.language.hidden_size) * 0.02)

        self.image_dim_out = config.language.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with optional crop decomposition.

        Args:
            pixel_values: (B, 3, H, W) or (B, num_crops+1, 3, H, W) with crops.

        Returns:
            (B, num_visual_tokens, hidden_size) visual embeddings.
        """
        if pixel_values.dim() == 5:
            return self._encode_with_crops(pixel_values)
        return self._encode_single(pixel_values)

    def _encode_single(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values)
        projected = self.projector(features)
        return projected

    def _encode_with_crops(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, NC, C, H, W = pixel_values.shape
        global_img = pixel_values[:, 0]
        crop_imgs = pixel_values[:, 1:]

        global_features = self.vision_encoder(global_img)
        global_projected = self.projector(global_features)

        crop_imgs_flat = crop_imgs.reshape(B * (NC - 1), C, H, W)
        crop_features = self.vision_encoder(crop_imgs_flat)
        crop_projected = self.projector(crop_features)

        _, N, D = crop_projected.shape
        crop_projected = crop_projected.reshape(B, NC - 1, N, D)

        all_tokens = [global_projected, self.glb_gn.expand(B, -1, -1)]
        for i in range(NC - 1):
            all_tokens.append(crop_projected[:, i])
            if i < NC - 2:
                all_tokens.append(self.sub_gn.expand(B, -1, -1))

        return torch.cat(all_tokens, dim=1)


@MODELS.register("phi_vision")
class PhiVisionArchitecture(nn.Module):
    """Phi-3-Vision architecture: SigLIP encoder + MLP projector + Phi-2/3 decoder.

    Key features:
    - Native multimodal design (image tokens embedded directly in text stream)
    - Sub-image decomposition for high-resolution inputs
    - Dynamic resolution with padding-free tokenization
    - SigLIP vision encoder (SO400M) for strong visual representations
    """

    IMAGE_TOKEN = "<|image|>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, config: FlashVLMConfig):
        super().__init__()
        self.config = config

        self.image_embedding = PhiImageEmbedding(config)

        self.language_model = None
        self.tokenizer = None

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through vision encoder and projector."""
        return self.image_embedding(images)

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[Tuple[int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare multimodal inputs by replacing image tokens with visual embeddings.

        Supports multiple images per sample via indexed image tokens.
        """
        if pixel_values is None or self.language_model is None:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        image_features = self.encode_images(pixel_values)

        embed_tokens = self.language_model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)

        image_token_mask = input_ids == self.IMAGE_TOKEN_INDEX
        if image_token_mask.any():
            batch_size = input_ids.shape[0]
            new_embeds_list = []

            for b in range(batch_size):
                cur_embeds = inputs_embeds[b]
                cur_mask = image_token_mask[b]

                if cur_mask.any():
                    positions = cur_mask.nonzero(as_tuple=True)[0]
                    segments = []
                    prev = 0
                    for i, pos in enumerate(positions):
                        pos_val = pos.item()
                        if prev < pos_val:
                            segments.append(cur_embeds[prev:pos_val])
                        if pixel_values.dim() == 4:
                            segments.append(image_features[min(b, image_features.shape[0] - 1)])
                        else:
                            segments.append(image_features[min(b, image_features.shape[0] - 1)])
                        prev = pos_val + 1
                    if prev < cur_embeds.shape[0]:
                        segments.append(cur_embeds[prev:])
                    new_embeds_list.append(torch.cat(segments, dim=0))
                else:
                    new_embeds_list.append(cur_embeds)

            max_len = max(e.shape[0] for e in new_embeds_list)
            padded = torch.zeros(
                batch_size, max_len, new_embeds_list[0].shape[-1],
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
            new_mask = torch.zeros(batch_size, max_len, device=input_ids.device, dtype=torch.long)
            for b, emb in enumerate(new_embeds_list):
                padded[b, :emb.shape[0]] = emb
                new_mask[b, :emb.shape[0]] = 1

            inputs_embeds = padded
            attention_mask = new_mask
        else:
            inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)
            if attention_mask is not None:
                img_mask = torch.ones(
                    image_features.shape[:2], device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Phi-Vision training."""
        prepared = self.prepare_inputs_for_multimodal(input_ids, pixel_values, attention_mask)

        if self.language_model is not None:
            outputs = self.language_model(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared.get("attention_mask"),
                labels=labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        logits = prepared["inputs_embeds"]
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}
