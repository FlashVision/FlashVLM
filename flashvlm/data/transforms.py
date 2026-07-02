"""Image transforms and augmentation for VLM training and inference."""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class VLMTransform:
    """Standard VLM image transform with normalization for CLIP-style models."""

    def __init__(
        self,
        image_size: int = 336,
        mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        augment: bool = False,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.8, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, image) -> torch.Tensor:
        return self.transform(image)


class DynamicResolutionTransform:
    """Transform that preserves aspect ratio with padding."""

    def __init__(
        self,
        max_size: int = 448,
        patch_size: int = 14,
        mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.max_size = max_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std

    def __call__(self, image) -> torch.Tensor:
        w, h = image.size
        scale = self.max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        new_w = (new_w // self.patch_size) * self.patch_size
        new_h = (new_h // self.patch_size) * self.patch_size

        image = TF.resize(
            image,
            (new_h, new_w),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        padded = torch.zeros(3, self.max_size, self.max_size)
        padded[:, :new_h, :new_w] = image
        return padded


class MultiCropTransform:
    """Multi-crop augmentation for VLM pre-training."""

    def __init__(
        self,
        image_size: int = 336,
        num_crops: int = 5,
        crop_scales: tuple[float, float] = (0.4, 1.0),
    ):
        self.global_transform = VLMTransform(image_size=image_size, augment=True)
        self.local_transforms = [
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size // 2,
                        scale=crop_scales,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
            for _ in range(num_crops)
        ]

    def __call__(self, image) -> dict:
        global_view = self.global_transform(image)
        local_views = [t(image) for t in self.local_transforms]
        return {"global": global_view, "locals": torch.stack(local_views)}


def build_transform(
    image_size: int = 336,
    augment: bool = False,
    dynamic_resolution: bool = False,
) -> VLMTransform:
    """Build an image transform based on configuration.

    Args:
        image_size: Target image size.
        augment: Whether to apply data augmentation.
        dynamic_resolution: Use dynamic resolution with padding.

    Returns:
        Transform callable.
    """
    if dynamic_resolution:
        return DynamicResolutionTransform(max_size=image_size)
    return VLMTransform(image_size=image_size, augment=augment)
