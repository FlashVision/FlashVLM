"""Vision encoders for FlashVLM: CLIP, SigLIP, DINOv2."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

from flashvlm.cfg.config import VisionConfig
from flashvlm.registry import VISION_ENCODERS


class VisionEncoder(nn.Module):
    """Base vision encoder that wraps pretrained vision transformers."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.config.patch_size) ** 2

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])


@VISION_ENCODERS.register("clip")
class CLIPVisionEncoder(VisionEncoder):
    """CLIP ViT vision encoder using HuggingFace transformers."""

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        try:
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(
                config.encoder_name, torch_dtype=torch.float16
            )
        except Exception:
            self.model = self._build_vit(config)

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _build_vit(self, config: VisionConfig) -> nn.Module:
        """Fallback: build a simple ViT if pretrained model unavailable."""
        return nn.Sequential(
            nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size),
            nn.Flatten(2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "vision_model"):
            outputs = self.model(pixel_values=pixel_values)
            return outputs.last_hidden_state
        return self.model(pixel_values)


@VISION_ENCODERS.register("siglip")
class SigLIPVisionEncoder(VisionEncoder):
    """SigLIP vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        try:
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(
                config.encoder_name, torch_dtype=torch.float16
            )
        except Exception:
            self.model = nn.Sequential(
                nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size),
                nn.Flatten(2),
            )

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "vision_model"):
            outputs = self.model(pixel_values=pixel_values)
            return outputs.last_hidden_state
        x = self.model[0](pixel_values)
        x = self.model[1](x).transpose(1, 2)
        return x


@VISION_ENCODERS.register("dinov2")
class DINOv2VisionEncoder(VisionEncoder):
    """DINOv2 vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        except Exception:
            self.model = nn.Sequential(
                nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size),
                nn.Flatten(2),
            )

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(pixel_values)
            if isinstance(features, dict):
                return features["x_norm_patchtokens"]
            return features
        x = self.model[0](pixel_values)
        return self.model[1](x).transpose(1, 2)

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_vision_encoder(config: VisionConfig) -> VisionEncoder:
    """Factory function to build a vision encoder from config."""
    encoder_name = config.encoder_name.lower()
    if "siglip" in encoder_name:
        return SigLIPVisionEncoder(config)
    elif "dinov2" in encoder_name or "dino" in encoder_name:
        return DINOv2VisionEncoder(config)
    else:
        return CLIPVisionEncoder(config)
