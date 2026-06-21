"""I/O utilities for FlashVLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image


def load_image(
    path: Union[str, Path],
    size: Optional[tuple[int, int]] = None,
    mode: str = "RGB",
) -> Image.Image:
    """Load an image from disk with optional resizing.

    Args:
        path: Path to the image file.
        size: Optional (width, height) to resize to.
        mode: Color mode (RGB, L, RGBA).

    Returns:
        PIL Image object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert(mode)
    if size is not None:
        image = image.resize(size, Image.BICUBIC)
    return image


def save_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """Save an image to disk.

    Args:
        image: Image as PIL, numpy array, or torch tensor.
        path: Output file path.
        quality: JPEG quality (1-100).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.dim() == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    save_kwargs = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality

    image.save(path, **save_kwargs)


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data as JSON.

    Args:
        data: Data to serialize.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_checkpoint(
    path: Union[str, Path], device: str = "cpu"
) -> Dict[str, Any]:
    """Load a model checkpoint.

    Args:
        path: Path to checkpoint file (.pt, .bin, .safetensors).
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary or state dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device=device)

    return torch.load(path, map_location=device, weights_only=True)


def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    path: Union[str, Path],
    use_safetensors: bool = True,
) -> None:
    """Save model weights.

    Args:
        state_dict: Model state dictionary.
        path: Output path.
        use_safetensors: Use safetensors format if available.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_safetensors:
        try:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))
            return
        except ImportError:
            pass

    torch.save(state_dict, path)
