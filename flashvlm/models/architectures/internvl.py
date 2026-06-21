"""InternVL-style architecture with dynamic resolution and pixel shuffle."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvlm.cfg.config import FlashVLMConfig
from flashvlm.models.vision_encoder import build_vision_encoder
from flashvlm.models.projector import MLPProjector
from flashvlm.registry import MODELS


class PixelShuffle(nn.Module):
    """Pixel shuffle downsampling for reducing visual token count."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape spatial tokens via pixel shuffle.

        Input: (B, H*W, C)  ->  Output: (B, H*W/scale^2, C*scale^2)
        """
        batch_size, num_tokens, channels = x.shape
        h = w = int(math.sqrt(num_tokens))

        x = x.view(batch_size, h, w, channels)
        sf = self.scale_factor
        x = x.view(batch_size, h // sf, sf, w // sf, sf, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, (h // sf) * (w // sf), channels * sf * sf)
        return x


class DynamicResolutionModule(nn.Module):
    """Handle dynamic image resolutions by splitting into tiles."""

    def __init__(self, base_size: int = 448, max_tiles: int = 12):
        super().__init__()
        self.base_size = base_size
        self.max_tiles = max_tiles

    def compute_tile_layout(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Compute optimal tile grid for a given image size."""
        h, w = image_size
        aspect_ratio = w / h

        best_layout = (1, 1)
        best_waste = float("inf")

        for rows in range(1, self.max_tiles + 1):
            for cols in range(1, self.max_tiles + 1):
                if rows * cols > self.max_tiles:
                    break
                tile_aspect = cols / rows
                waste = abs(tile_aspect - aspect_ratio)
                if waste < best_waste:
                    best_waste = waste
                    best_layout = (rows, cols)

        return best_layout

    def split_into_tiles(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Split an image tensor into tiles of base_size."""
        _, _, h, w = image.shape
        rows, cols = self.compute_tile_layout((h, w))
        tile_h = h // rows
        tile_w = w // cols

        tiles = []
        for r in range(rows):
            for c in range(cols):
                tile = image[:, :, r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
                tile = F.interpolate(
                    tile, size=(self.base_size, self.base_size), mode="bicubic", align_corners=False
                )
                tiles.append(tile)

        thumbnail = F.interpolate(
            image, size=(self.base_size, self.base_size), mode="bicubic", align_corners=False
        )
        tiles.insert(0, thumbnail)

        return tiles


@MODELS.register("internvl")
class InternVLArchitecture(nn.Module):
    """InternVL-style architecture.

    Features: Dynamic resolution with pixel shuffle, InternViT vision encoder,
    and MLP projection to language model.
    """

    def __init__(self, config: FlashVLMConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = build_vision_encoder(config.vision)

        self.pixel_shuffle = PixelShuffle(scale_factor=2)

        shuffled_dim = config.vision.hidden_size * 4
        proj_cfg = config.projector
        proj_cfg.type = "mlp"
        proj_cfg.input_dim = shuffled_dim
        proj_cfg.output_dim = config.language.hidden_size
        proj_cfg.num_layers = 2
        self.mlp_projector = MLPProjector(proj_cfg)

        self.dynamic_resolution = DynamicResolutionModule(
            base_size=config.vision.image_size,
            max_tiles=12,
        )

        self.language_model = None

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        use_dynamic_resolution: bool = True,
    ) -> torch.Tensor:
        """Encode images with optional dynamic resolution."""
        if use_dynamic_resolution and pixel_values.shape[-1] > self.config.vision.image_size:
            tiles = self.dynamic_resolution.split_into_tiles(pixel_values)
            tile_features = []
            for tile in tiles:
                feat = self.vision_encoder(tile)
                feat = self.pixel_shuffle(feat)
                tile_features.append(feat)
            visual_tokens = torch.cat(tile_features, dim=1)
        else:
            features = self.vision_encoder(pixel_values)
            visual_tokens = self.pixel_shuffle(features)

        projected = self.mlp_projector(visual_tokens)
        return projected

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for InternVL."""
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
        else:
            visual_tokens = None

        if self.language_model is not None:
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
            if visual_tokens is not None:
                inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
                if attention_mask is not None:
                    vis_mask = torch.ones(
                        visual_tokens.shape[:2],
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            else:
                inputs_embeds = text_embeds

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        logits = visual_tokens if visual_tokens is not None else torch.zeros(
            input_ids.shape[0], 1, self.config.language.hidden_size, device=input_ids.device
        )
        return {"loss": None, "logits": logits}
