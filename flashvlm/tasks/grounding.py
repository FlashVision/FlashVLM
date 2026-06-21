"""Visual Grounding / Referring Expression Comprehension task."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from flashvlm.registry import TASKS


@TASKS.register("grounding")
class GroundingTask:
    """Visual grounding: locate objects in images from text descriptions.

    Outputs normalized bounding boxes [x1, y1, x2, y2] in [0, 1] range.
    """

    PROMPT_TEMPLATES = {
        "default": "Locate the following in the image: {expression}\nBounding box:",
        "point": "Point to {expression} in the image.\nCoordinates:",
        "region": "Identify the region containing: {expression}\nRegion [x1, y1, x2, y2]:",
    }

    def __init__(
        self,
        model: Any,
        prompt_style: str = "default",
        max_new_tokens: int = 64,
        temperature: float = 0.1,
    ):
        self.model = model
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def ground(
        self,
        image: str | Path | Image.Image,
        expression: str,
        **kwargs: Any,
    ) -> list[float] | None:
        """Locate an object described by a text expression.

        Args:
            image: Input image.
            expression: Text description of the object to locate.

        Returns:
            Normalized bounding box [x1, y1, x2, y2] or None if parsing fails.
        """
        prompt = self._format_prompt(expression)
        response = self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        return self._parse_bbox(response)

    def batch_ground(
        self,
        images: list[str | Path | Image.Image],
        expressions: list[str],
        **kwargs: Any,
    ) -> list[list[float] | None]:
        """Ground multiple expressions in their respective images."""
        return [self.ground(img, expr, **kwargs) for img, expr in zip(images, expressions)]

    def _format_prompt(self, expression: str) -> str:
        template = self.PROMPT_TEMPLATES.get(self.prompt_style, self.PROMPT_TEMPLATES["default"])
        return template.format(expression=expression)

    def _parse_bbox(self, response: str) -> list[float] | None:
        """Parse bounding box coordinates from model output."""
        patterns = [
            r"\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]?",
            r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                coords = [float(match.group(i)) for i in range(1, 5)]
                if all(c > 1.0 for c in coords):
                    coords = [c / 1000.0 for c in coords]
                coords = [max(0.0, min(1.0, c)) for c in coords]
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    return coords

        return None

    def evaluate(
        self,
        predictions: list[list[float] | None],
        ground_truths: list[list[float]],
        iou_threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate grounding accuracy using IoU.

        Args:
            predictions: List of predicted bounding boxes.
            ground_truths: List of ground truth bounding boxes.
            iou_threshold: IoU threshold for a correct prediction.

        Returns:
            Dictionary with accuracy and mean IoU.
        """
        correct = 0
        total_iou = 0.0
        valid = 0

        for pred, gt in zip(predictions, ground_truths):
            if pred is None:
                continue
            iou = self._compute_iou(pred, gt)
            total_iou += iou
            valid += 1
            if iou >= iou_threshold:
                correct += 1

        total = len(ground_truths)
        return {
            f"accuracy@{iou_threshold}": correct / max(total, 1),
            "mean_iou": total_iou / max(valid, 1),
            "parse_rate": valid / max(total, 1),
        }

    @staticmethod
    def _compute_iou(box1: list[float], box2: list[float]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)
