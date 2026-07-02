"""Multimodal tokenizer wrapper for VLM training and inference."""

from __future__ import annotations

from typing import Any

import torch


class MultimodalTokenizer:
    """Wrapper around HuggingFace tokenizers with multimodal token support.

    Handles special tokens for images, bounding boxes, and other visual
    elements that VLMs need to process.
    """

    SPECIAL_TOKENS = {
        "image_token": "<image>",
        "image_start": "<img>",
        "image_end": "</img>",
        "bbox_start": "<box>",
        "bbox_end": "</box>",
        "ref_start": "<ref>",
        "ref_end": "</ref>",
        "ocr_start": "<ocr>",
        "ocr_end": "</ocr>",
    }

    def __init__(
        self,
        tokenizer_name_or_path: str,
        add_visual_tokens: bool = True,
        max_length: int = 2048,
    ):
        self.max_length = max_length
        self._tokenizer = self._load_tokenizer(tokenizer_name_or_path)

        if add_visual_tokens:
            self._add_special_tokens()

    def _load_tokenizer(self, name_or_path: str) -> Any:
        """Load a HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception:
            return self._create_simple_tokenizer()

    def _create_simple_tokenizer(self) -> SimpleTokenizer:
        """Fallback: simple character-level tokenizer for testing."""
        return SimpleTokenizer()

    def _add_special_tokens(self) -> None:
        """Add multimodal special tokens to the tokenizer."""
        new_tokens = list(self.SPECIAL_TOKENS.values())
        if hasattr(self._tokenizer, "add_special_tokens"):
            self._tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    @property
    def vocab_size(self) -> int:
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return len(self._tokenizer)

    @property
    def pad_token_id(self) -> int:
        if hasattr(self._tokenizer, "pad_token_id"):
            return self._tokenizer.pad_token_id or 0
        return 0

    @property
    def eos_token_id(self) -> int:
        if hasattr(self._tokenizer, "eos_token_id"):
            return self._tokenizer.eos_token_id or 2
        return 2

    @property
    def image_token_id(self) -> int:
        """Get the token ID for the image placeholder."""
        token = self.SPECIAL_TOKENS["image_token"]
        if hasattr(self._tokenizer, "convert_tokens_to_ids"):
            return self._tokenizer.convert_tokens_to_ids(token)
        return -200

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encode text with proper handling of visual tokens.

        Args:
            text: Input text with optional visual token placeholders.
            max_length: Override max sequence length.
            padding: Padding strategy.
            truncation: Whether to truncate.
            return_tensors: Output format.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        max_length = max_length or self.max_length

        if hasattr(self._tokenizer, "__call__"):
            return self._tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
            )

        return self._tokenizer.encode(text, max_length=max_length, return_tensors=return_tensors)

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return "".join(chr(min(t, 127)) for t in token_ids if t >= 0)

    def format_multimodal_prompt(
        self,
        text: str,
        num_images: int = 1,
        system_prompt: str | None = None,
    ) -> str:
        """Format a multimodal prompt with image placeholders.

        Args:
            text: User text prompt.
            num_images: Number of images to insert.
            system_prompt: Optional system instruction.

        Returns:
            Formatted prompt string.
        """
        image_tokens = " ".join([self.SPECIAL_TOKENS["image_token"]] * num_images)
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        parts.append(f"{image_tokens}\nUser: {text}\nAssistant:")
        return "\n".join(parts)

    def get_num_image_tokens(self, image_size: int = 336, patch_size: int = 14) -> int:
        """Calculate the number of visual tokens for an image."""
        return (image_size // patch_size) ** 2


class SimpleTokenizer:
    """Minimal tokenizer for testing without HuggingFace."""

    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __len__(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        text: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        token_ids = [ord(c) % self._vocab_size for c in text]
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length":
            pad_len = max_length - len(token_ids)
            token_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def encode(self, text: str, **kwargs) -> torch.Tensor:
        result = self(text, **kwargs)
        return result["input_ids"]

    def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        chars = []
        for t in token_ids:
            if skip_special_tokens and t in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(chr(min(max(t, 32), 126)))
        return "".join(chars)

    def add_special_tokens(self, tokens_dict: dict) -> None:
        pass

    def convert_tokens_to_ids(self, token: str) -> int:
        return hash(token) % self._vocab_size
