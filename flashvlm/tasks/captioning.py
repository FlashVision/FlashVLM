"""Image Captioning task module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from flashvlm.registry import TASKS


@TASKS.register("captioning")
class CaptioningTask:
    """Image captioning with configurable detail levels and styles."""

    PROMPT_TEMPLATES = {
        "brief": "Describe this image briefly.",
        "detailed": (
            "Describe this image in detail, including all visible "
            "objects, their positions, colors, and any activities "
            "or interactions."
        ),
        "creative": "Write a creative and engaging caption for this image.",
        "technical": (
            "Provide a technical description of this image, noting "
            "specific objects, their attributes, and spatial "
            "relationships."
        ),
        "accessibility": (
            "Describe this image for someone who cannot see it, "
            "focusing on the most important visual information."
        ),
    }

    def __init__(
        self,
        model: Any,
        style: str = "detailed",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.model = model
        self.style = style
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def caption(
        self,
        image: str | Path | Image.Image,
        style: str | None = None,
        custom_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a caption for an image.

        Args:
            image: Input image.
            style: Caption style (brief, detailed, creative, technical, accessibility).
            custom_prompt: Override the default prompt.

        Returns:
            Generated caption string.
        """
        style = style or self.style
        prompt = custom_prompt or self.PROMPT_TEMPLATES.get(
            style, self.PROMPT_TEMPLATES["detailed"]
        )

        caption = self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        return self._postprocess(caption)

    def batch_caption(
        self,
        images: list[str | Path | Image.Image],
        style: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate captions for multiple images."""
        return [self.caption(img, style=style, **kwargs) for img in images]

    def caption_with_context(
        self,
        image: str | Path | Image.Image,
        context: str,
        **kwargs: Any,
    ) -> str:
        """Generate a contextual caption.

        Args:
            image: Input image.
            context: Additional context (e.g., "This is a medical image").

        Returns:
            Contextual caption.
        """
        prompt = f"Context: {context}\nDescribe this image based on the given context."
        return self.caption(image, custom_prompt=prompt, **kwargs)

    def _postprocess(self, caption: str) -> str:
        """Clean caption output."""
        caption = caption.strip()
        if caption and caption[-1] not in ".!?":
            caption += "."
        return caption

    def evaluate(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> dict[str, float]:
        """Evaluate captions using n-gram metrics.

        Computes simplified BLEU-4 and CIDEr-like scores.
        """
        from flashvlm.analytics.metrics import compute_bleu, compute_cider

        bleu_score = compute_bleu(predictions, references)
        cider_score = compute_cider(predictions, references)

        return {"bleu4": bleu_score, "cider": cider_score}
