"""Document/Scene OCR via Vision-Language Model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from flashvlm.registry import TASKS


@TASKS.register("ocr")
class OCRTask:
    """OCR task using VLMs for text extraction from images.

    Handles documents, scene text, handwriting, and structured content.
    """

    PROMPT_TEMPLATES = {
        "extract_all": (
            "Extract all text visible in this image,"
            " preserving the layout as much as possible."
        ),
        "document": "Read the document in this image and output its text content in order.",
        "scene": "Identify and transcribe any text visible in this scene image.",
        "handwriting": "Read the handwritten text in this image.",
        "structured": (
            "Extract the structured information (tables, forms,"
            " key-value pairs) from this document image."
        ),
        "specific_region": "Read the text in the {region} of this image.",
    }

    def __init__(
        self,
        model: Any,
        mode: str = "extract_all",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model = model
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def extract_text(
        self,
        image: str | Path | Image.Image,
        mode: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Extract text from an image.

        Args:
            image: Input image.
            mode: OCR mode (extract_all, document, scene, handwriting, structured).
            region: Specific region description (for specific_region mode).

        Returns:
            Extracted text.
        """
        mode = mode or self.mode
        if mode == "specific_region" and region:
            prompt = self.PROMPT_TEMPLATES["specific_region"].format(region=region)
        else:
            prompt = self.PROMPT_TEMPLATES.get(mode, self.PROMPT_TEMPLATES["extract_all"])

        result = self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        return result.strip()

    def extract_structured(
        self,
        image: str | Path | Image.Image,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract structured data (key-value pairs, tables) from a document.

        Args:
            image: Document image.

        Returns:
            Dictionary with extracted structured information.
        """
        prompt = (
            "Extract all key-value pairs and tables from this document. "
            "Format as:\nKey: Value\n\nFor tables, use | as column separator."
        )
        raw_text = self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            **kwargs,
        )
        return self._parse_structured(raw_text)

    def _parse_structured(self, text: str) -> dict[str, Any]:
        """Parse structured output into a dictionary."""
        result: dict[str, Any] = {"key_values": {}, "tables": [], "raw_text": text}

        lines = text.strip().split("\n")
        current_table: list[list[str]] = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_table:
                    result["tables"].append(current_table)
                    current_table = []
                continue

            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells:
                    current_table.append(cells)
            elif ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    result["key_values"][key] = value

        if current_table:
            result["tables"].append(current_table)

        return result

    def batch_extract(
        self,
        images: list[str | Path | Image.Image],
        mode: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Extract text from multiple images."""
        return [self.extract_text(img, mode=mode, **kwargs) for img in images]

    def evaluate(
        self,
        predictions: list[str],
        ground_truths: list[str],
    ) -> dict[str, float]:
        """Evaluate OCR accuracy using character and word level metrics."""
        total_char_correct = 0
        total_chars = 0
        total_word_correct = 0
        total_words = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_clean = pred.lower().strip()
            gt_clean = gt.lower().strip()

            total_chars += len(gt_clean)
            for pc, gc in zip(pred_clean, gt_clean):
                if pc == gc:
                    total_char_correct += 1

            pred_words = set(pred_clean.split())
            gt_words = set(gt_clean.split())
            total_words += len(gt_words)
            total_word_correct += len(pred_words & gt_words)

        return {
            "char_accuracy": total_char_correct / max(total_chars, 1),
            "word_accuracy": total_word_correct / max(total_words, 1),
        }
