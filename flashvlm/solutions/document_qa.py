"""Document understanding and QA solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class DocumentQA:
    """Document question answering with support for multi-page documents.

    Combines OCR, layout understanding, and VQA capabilities.
    """

    def __init__(
        self,
        model_name: str = "llava-v1.5-7b",
        device: str = "auto",
        max_tokens: int = 512,
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

    def ask(
        self,
        document: str | Path | Image.Image | list[Image.Image],
        question: str,
        **kwargs: Any,
    ) -> str:
        """Ask a question about a document.

        Args:
            document: Document image(s) or path.
            question: Question about the document content.

        Returns:
            Answer string.
        """
        if isinstance(document, list):
            return self._multi_page_qa(document, question, **kwargs)

        prompt = (
            f"You are analyzing a document image. "
            f"Answer the following question based on the document content.\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        if self.model is not None:
            return self.model.generate(
                prompt=prompt,
                image=document,
                max_new_tokens=self.max_tokens,
                temperature=0.1,
                **kwargs,
            )
        return f"[Model not loaded. Question: {question}]"

    def extract_info(
        self,
        document: str | Path | Image.Image,
        fields: list[str],
        **kwargs: Any,
    ) -> dict[str, str]:
        """Extract specific fields from a document.

        Args:
            document: Document image.
            fields: List of field names to extract.

        Returns:
            Dictionary mapping field names to extracted values.
        """
        fields_str = ", ".join(fields)
        prompt = (
            f"Extract the following information from this document: {fields_str}\n\n"
            f"Format your response as:\nField: Value\n\nExtracted information:"
        )

        if self.model is not None:
            response = self.model.generate(
                prompt=prompt,
                image=document,
                max_new_tokens=self.max_tokens,
                temperature=0.1,
                **kwargs,
            )
            return self._parse_fields(response, fields)
        return {field: "" for field in fields}

    def summarize(
        self,
        document: str | Path | Image.Image,
        max_length: int = 200,
        **kwargs: Any,
    ) -> str:
        """Generate a summary of a document.

        Args:
            document: Document image.
            max_length: Maximum summary length in tokens.

        Returns:
            Summary text.
        """
        prompt = (
            "Provide a concise summary of this document, "
            "covering the main points and key information."
        )

        if self.model is not None:
            return self.model.generate(
                prompt=prompt,
                image=document,
                max_new_tokens=max_length,
                temperature=0.3,
                **kwargs,
            )
        return "[Model not loaded]"

    def _multi_page_qa(
        self,
        pages: list[Image.Image],
        question: str,
        **kwargs: Any,
    ) -> str:
        """Handle multi-page document QA by processing pages sequentially."""
        page_responses = []
        for i, page in enumerate(pages):
            prompt = (
                f"This is page {i + 1} of a document. "
                f"Extract relevant information for the question: {question}"
            )
            if self.model is not None:
                resp = self.model.generate(
                    prompt=prompt,
                    image=page,
                    max_new_tokens=200,
                    temperature=0.1,
                    **kwargs,
                )
                page_responses.append(f"Page {i + 1}: {resp}")

        context = "\n".join(page_responses)
        final_prompt = (
            f"Based on the following extracted information from a multi-page document:\n\n"
            f"{context}\n\nQuestion: {question}\nFinal Answer:"
        )

        if self.model is not None:
            return self.model.generate(
                prompt=final_prompt,
                max_new_tokens=self.max_tokens,
                temperature=0.1,
            )
        return "[Model not loaded]"

    def _parse_fields(self, response: str, fields: list[str]) -> dict[str, str]:
        """Parse field-value pairs from model response."""
        result = {field: "" for field in fields}
        for line in response.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                for field in fields:
                    if field.lower() in key.lower():
                        result[field] = value
                        break
        return result
