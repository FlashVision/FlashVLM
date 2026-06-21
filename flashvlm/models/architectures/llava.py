"""LLaVA-style architecture: CLIP encoder + MLP projector + LLaMA decoder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from flashvlm.cfg.config import FlashVLMConfig, VisionConfig, ProjectorConfig
from flashvlm.models.vision_encoder import CLIPVisionEncoder
from flashvlm.models.projector import MLPProjector
from flashvlm.registry import MODELS


@MODELS.register("llava")
class LLaVAArchitecture(nn.Module):
    """LLaVA (Large Language and Vision Assistant) architecture.

    Architecture: CLIP ViT-L/14 -> 2-layer MLP -> LLaMA-2 7B
    """

    IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, config: FlashVLMConfig):
        super().__init__()
        self.config = config

        vision_cfg = config.vision
        vision_cfg.encoder_name = vision_cfg.encoder_name or "openai/clip-vit-large-patch14-336"
        self.vision_tower = CLIPVisionEncoder(vision_cfg)

        proj_cfg = config.projector
        proj_cfg.type = "mlp"
        proj_cfg.input_dim = vision_cfg.hidden_size
        proj_cfg.output_dim = config.language.hidden_size
        proj_cfg.num_layers = 2
        self.mm_projector = MLPProjector(proj_cfg)

        self.language_model = None
        self.tokenizer = None

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through vision tower and projector."""
        vision_features = self.vision_tower(images)
        image_features = self.mm_projector(vision_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare multimodal inputs by replacing image tokens with visual embeddings."""
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
                cur_input_embeds = inputs_embeds[b]
                cur_image_mask = image_token_mask[b]
                if cur_image_mask.any():
                    image_idx = cur_image_mask.nonzero(as_tuple=True)[0][0]
                    before = cur_input_embeds[:image_idx]
                    after = cur_input_embeds[image_idx + 1:]
                    new_embeds = torch.cat([before, image_features[b], after], dim=0)
                    new_embeds_list.append(new_embeds)
                else:
                    new_embeds_list.append(cur_input_embeds)

            max_len = max(e.shape[0] for e in new_embeds_list)
            padded_embeds = torch.zeros(
                batch_size, max_len, new_embeds_list[0].shape[-1],
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
            new_attention_mask = torch.zeros(
                batch_size, max_len, device=input_ids.device, dtype=torch.long
            )
            for b, embeds in enumerate(new_embeds_list):
                padded_embeds[b, :embeds.shape[0]] = embeds
                new_attention_mask[b, :embeds.shape[0]] = 1

            inputs_embeds = padded_embeds
            attention_mask = new_attention_mask
        else:
            inputs_embeds = torch.cat(
                [image_features, inputs_embeds], dim=1
            )
            if attention_mask is not None:
                img_mask = torch.ones(
                    image_features.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype
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
        """Forward pass for training."""
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
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return {"loss": loss, "logits": logits}
