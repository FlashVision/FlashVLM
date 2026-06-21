"""Inference predictor for FlashVLM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image

from flashvlm.cfg.config import FlashVLMConfig


class Predictor:
    """High-level inference interface for FlashVLM models.

    Handles image preprocessing, prompt formatting, batched inference,
    and output post-processing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FlashVLMConfig] = None,
        device: str = "auto",
    ):
        self.model = model
        self.config = config or FlashVLMConfig()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """Run inference on a single image with a text prompt.

        Args:
            image: Input image (path, PIL Image, or tensor).
            prompt: Text prompt for the model.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text response.
        """
        return self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> List[str]:
        """Run batched inference on multiple image-prompt pairs.

        Args:
            images: List of input images.
            prompts: List of text prompts (must match images in length).
            max_new_tokens: Maximum tokens to generate per sample.

        Returns:
            List of generated text responses.
        """
        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match prompts ({len(prompts)})"
            )

        results = []
        for image, prompt in zip(images, prompts):
            result = self.predict(
                image=image, prompt=prompt, max_new_tokens=max_new_tokens, **kwargs
            )
            results.append(result)

        return results

    @torch.no_grad()
    def vqa(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Visual Question Answering shortcut.

        Args:
            image: Input image.
            question: Question about the image.
            max_new_tokens: Max answer length.

        Returns:
            The model's answer.
        """
        return self.model.ask(question, image=image, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def caption(
        self,
        image: Union[str, Path, Image.Image],
        max_new_tokens: int = 100,
    ) -> str:
        """Generate a caption for an image.

        Args:
            image: Input image.
            max_new_tokens: Max caption length.

        Returns:
            Generated caption.
        """
        return self.model.caption(image, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def get_visual_embeddings(
        self, image: Union[str, Path, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """Extract visual embeddings without text generation.

        Args:
            image: Input image.

        Returns:
            Visual embedding tensor.
        """
        return self.model.encode_image(image)
