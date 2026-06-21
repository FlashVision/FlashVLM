"""Qwen-VL style architecture with visual tokenization and positional encoding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvlm.cfg.config import FlashVLMConfig
from flashvlm.models.vision_encoder import build_vision_encoder
from flashvlm.models.projector import QFormerProjector
from flashvlm.registry import MODELS


class VisualPositionEncoding(nn.Module):
    """2D positional encoding for visual tokens with bounding box support."""

    def __init__(self, hidden_size: int, max_patches: int = 1024):
        super().__init__()
        self.row_embed = nn.Embedding(max_patches, hidden_size // 2)
        self.col_embed = nn.Embedding(max_patches, hidden_size // 2)
        self.spatial_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_tokens: torch.Tensor, image_size: Tuple[int, int] = (24, 24)) -> torch.Tensor:
        h, w = image_size
        batch_size, num_tokens, hidden = visual_tokens.shape

        row_ids = torch.arange(h, device=visual_tokens.device).unsqueeze(1).expand(h, w).reshape(-1)
        col_ids = torch.arange(w, device=visual_tokens.device).unsqueeze(0).expand(h, w).reshape(-1)

        if num_tokens > h * w:
            row_ids = row_ids[:num_tokens]
            col_ids = col_ids[:num_tokens]
        elif num_tokens < h * w:
            row_ids = row_ids[:num_tokens]
            col_ids = col_ids[:num_tokens]

        row_emb = self.row_embed(row_ids)
        col_emb = self.col_embed(col_ids)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        pos_emb = self.spatial_proj(pos_emb)

        return visual_tokens + pos_emb.unsqueeze(0).expand(batch_size, -1, -1)


@MODELS.register("qwen_vl")
class QwenVLArchitecture(nn.Module):
    """Qwen-VL style architecture.

    Features: Visual tokenizer with spatial position encoding,
    cross-attention resampler, and interleaved image-text format.
    """

    def __init__(self, config: FlashVLMConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = build_vision_encoder(config.vision)

        proj_cfg = config.projector
        proj_cfg.type = "qformer"
        proj_cfg.input_dim = config.vision.hidden_size
        proj_cfg.output_dim = config.language.hidden_size
        self.visual_resampler = QFormerProjector(proj_cfg)

        self.position_encoding = VisualPositionEncoding(
            config.language.hidden_size,
            max_patches=(config.vision.image_size // config.vision.patch_size) ** 2,
        )

        self.visual_start_token = nn.Parameter(torch.randn(1, 1, config.language.hidden_size) * 0.02)
        self.visual_end_token = nn.Parameter(torch.randn(1, 1, config.language.hidden_size) * 0.02)
        self.language_model = None

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with spatial position encoding."""
        vision_features = self.vision_encoder(images)
        resampled = self.visual_resampler(vision_features)

        h = w = int(vision_features.shape[1] ** 0.5)
        positioned = self.position_encoding(resampled, (h, w))
        return positioned

    def wrap_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Wrap visual tokens with start/end markers."""
        batch_size = visual_tokens.shape[0]
        start = self.visual_start_token.expand(batch_size, -1, -1)
        end = self.visual_end_token.expand(batch_size, -1, -1)
        return torch.cat([start, visual_tokens, end], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Qwen-VL."""
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
            visual_tokens = self.wrap_visual_tokens(visual_tokens)
        else:
            visual_tokens = None

        if self.language_model is not None:
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
            if visual_tokens is not None:
                inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            else:
                inputs_embeds = text_embeds

            outputs = self.language_model(inputs_embeds=inputs_embeds, labels=labels)
            return {"loss": outputs.loss, "logits": outputs.logits}

        logits = visual_tokens if visual_tokens is not None else torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.config.language.hidden_size,
            device=input_ids.device,
        )
        return {"loss": None, "logits": logits}
