"""Comprehensive image analysis solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class ImageAnalyzer:
    """Comprehensive image analysis combining multiple VLM capabilities.

    Provides unified interface for captioning, object detection,
    scene understanding, and detailed visual analysis.
    """

    def __init__(
        self,
        model_name: str = "llava-v1.5-7b",
        device: str = "auto",
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from flashvlm.models.vlm import FlashVLM

            self.model = FlashVLM.from_pretrained(self.model_name, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    def analyze(
        self,
        image: str | Path | Image.Image,
        aspects: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Perform comprehensive image analysis.

        Args:
            image: Input image.
            aspects: Specific aspects to analyze. Defaults to all.
                Options: 'caption', 'objects', 'scene', 'colors', 'text',
                'emotions', 'actions', 'composition'

        Returns:
            Dictionary mapping aspect names to analysis results.
        """
        if aspects is None:
            aspects = ["caption", "objects", "scene", "colors", "actions"]

        results = {}
        for aspect in aspects:
            prompt = self._get_aspect_prompt(aspect)
            if self.model is not None:
                result = self.model.generate(
                    prompt=prompt,
                    image=image,
                    max_new_tokens=self.max_tokens // len(aspects),
                    temperature=0.3,
                    **kwargs,
                )
                results[aspect] = result.strip()
            else:
                results[aspect] = f"[Analysis for: {aspect}]"

        return results

    def full_analysis(
        self,
        image: str | Path | Image.Image,
        **kwargs: Any,
    ) -> str:
        """Generate a comprehensive, detailed analysis of an image.

        Args:
            image: Input image.

        Returns:
            Detailed analysis text covering all visual aspects.
        """
        prompt = (
            "Provide a comprehensive analysis of this image covering:\n"
            "1. Overall description and scene\n"
            "2. Key objects and their positions\n"
            "3. Colors, lighting, and atmosphere\n"
            "4. Any text or signage visible\n"
            "5. Actions or activities taking place\n"
            "6. Notable details or observations\n\n"
            "Detailed Analysis:"
        )

        if self.model is not None:
            return self.model.generate(
                prompt=prompt,
                image=image,
                max_new_tokens=self.max_tokens,
                temperature=0.5,
                **kwargs,
            )
        return "[Model not loaded for full analysis]"

    def compare(
        self,
        image1: str | Path | Image.Image,
        image2: str | Path | Image.Image,
        **kwargs: Any,
    ) -> str:
        """Compare two images and describe differences/similarities.

        Args:
            image1: First image.
            image2: Second image.

        Returns:
            Comparison analysis text.
        """
        analysis1 = self.analyze(image1, aspects=["caption", "objects"])
        analysis2 = self.analyze(image2, aspects=["caption", "objects"])

        comparison = (
            f"Image 1: {analysis1.get('caption', 'N/A')}\n"
            f"Image 2: {analysis2.get('caption', 'N/A')}\n"
        )
        return comparison

    def detect_objects(
        self,
        image: str | Path | Image.Image,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Detect and list objects in an image.

        Args:
            image: Input image.

        Returns:
            List of detected objects with descriptions.
        """
        prompt = (
            "List all distinct objects visible in this image. "
            "For each object, provide its name and approximate location "
            "(e.g., top-left, center, bottom-right).\n\n"
            "Objects:"
        )

        if self.model is not None:
            response = self.model.generate(
                prompt=prompt,
                image=image,
                max_new_tokens=300,
                temperature=0.2,
                **kwargs,
            )
            return self._parse_objects(response)
        return []

    def _get_aspect_prompt(self, aspect: str) -> str:
        """Get the prompt for a specific analysis aspect."""
        prompts = {
            "caption": "Describe this image in one detailed sentence.",
            "objects": "List all visible objects in this image.",
            "scene": "Describe the scene type and setting of this image.",
            "colors": "Describe the dominant colors and color palette in this image.",
            "text": "List any text or writing visible in this image.",
            "emotions": "What emotions or mood does this image convey?",
            "actions": "What actions or activities are taking place in this image?",
            "composition": "Describe the visual composition and layout of this image.",
        }
        return prompts.get(aspect, f"Analyze the {aspect} aspect of this image.")

    def _parse_objects(self, response: str) -> list[dict[str, Any]]:
        """Parse object list from model response."""
        objects = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("-•*0123456789. ")
            if line:
                objects.append({"name": line, "confidence": 1.0})
        return objects
