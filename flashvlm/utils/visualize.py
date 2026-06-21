"""Visualization utilities for VLM outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def draw_bbox(
    image: str | Path | Image.Image,
    boxes: list[list[float]],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    line_width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: Input image.
        boxes: List of [x1, y1, x2, y2] boxes (normalized 0-1 or pixel coords).
        labels: Optional text labels for each box.
        colors: Optional colors for each box.
        line_width: Box border width.
        font_size: Label font size.

    Returns:
        Image with drawn bounding boxes.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.copy()

    draw = ImageDraw.Draw(image)
    width, height = image.size

    default_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if all(0 <= c <= 1 for c in [x1, y1, x2, y2]):
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height

        color = colors[i] if colors else default_colors[i % len(default_colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if labels and i < len(labels):
            label = labels[i]
            text_bbox = draw.textbbox((x1, y1 - font_size - 4), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - font_size - 4), label, fill="white", font=font)

    return image


def visualize_attention(
    image: str | Path | Image.Image,
    attention_map: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay attention heatmap on an image.

    Args:
        image: Input image.
        attention_map: 2D attention weights (H, W) or (num_patches,).
        alpha: Blending factor (0=image only, 1=heatmap only).
        colormap: Matplotlib-compatible colormap name.

    Returns:
        Image with attention overlay.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    width, height = image.size

    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()

    if attention_map.ndim == 1:
        side = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map[: side * side].reshape(side, side)

    attn_min = attention_map.min()
    attn_max = attention_map.max()
    if attn_max > attn_min:
        attention_map = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attention_map = np.zeros_like(attention_map)

    heatmap = np.zeros((*attention_map.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (attention_map * 255).astype(np.uint8)
    heatmap[..., 1] = ((1 - np.abs(2 * attention_map - 1)) * 255).astype(np.uint8)
    heatmap[..., 2] = ((1 - attention_map) * 255).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap, mode="RGB")
    heatmap_img = heatmap_img.resize((width, height), Image.BICUBIC)

    blended = Image.blend(image, heatmap_img, alpha=alpha)
    return blended


def create_comparison_grid(
    images: list[Image.Image],
    captions: list[str] | None = None,
    cols: int = 4,
    cell_size: tuple[int, int] = (256, 256),
    padding: int = 8,
) -> Image.Image:
    """Create a grid of images for comparison.

    Args:
        images: List of images to display.
        captions: Optional captions for each image.
        cols: Number of columns.
        cell_size: Size of each cell (width, height).
        padding: Padding between cells.

    Returns:
        Grid image.
    """
    rows = (len(images) + cols - 1) // cols
    cell_w, cell_h = cell_size

    grid_w = cols * (cell_w + padding) + padding
    grid_h = rows * (cell_h + padding + (20 if captions else 0)) + padding

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding + (20 if captions else 0))

        resized = img.resize((cell_w, cell_h), Image.BICUBIC)
        grid.paste(resized, (x, y))

        if captions and idx < len(captions):
            text_y = y + cell_h + 2
            draw.text((x, text_y), captions[idx][:30], fill="black", font=font)

    return grid
