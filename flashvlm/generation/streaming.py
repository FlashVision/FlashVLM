"""Streaming text generation for real-time output."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Callable

import torch
import torch.nn as nn

from flashvlm.generation.sampler import combined_sample


class StreamingGenerator:
    """Token-by-token streaming generation with callback support."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: list[int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stop_tokens = stop_tokens or []

        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.stop_tokens.append(tokenizer.eos_token_id)

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> Generator[str, None, None]:
        """Generate tokens one at a time, yielding decoded text chunks.

        Args:
            input_ids: Initial input token IDs.
            attention_mask: Optional attention mask.
            inputs_embeds: Optional pre-computed embeddings.

        Yields:
            Decoded text for each new token.
        """
        device = input_ids.device
        generated_ids = input_ids.clone()
        past_key_values = None

        for step in range(self.max_new_tokens):
            if past_key_values is not None and hasattr(self.model, "forward"):
                model_inputs = {
                    "input_ids": generated_ids[:, -1:],
                    "attention_mask": torch.ones(
                        generated_ids.shape[0], generated_ids.shape[1], device=device
                    ),
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
            else:
                model_inputs = {
                    "input_ids": generated_ids,
                    "attention_mask": torch.ones_like(generated_ids),
                }
                if inputs_embeds is not None and step == 0:
                    model_inputs["inputs_embeds"] = inputs_embeds
                    del model_inputs["input_ids"]

            try:
                outputs = self.model(**model_inputs)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                    past_key_values = outputs.get("past_key_values")
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                    past_key_values = getattr(outputs, "past_key_values", None)
                else:
                    logits = outputs
            except (TypeError, AttributeError):
                break

            if logits is None:
                break

            next_logits = logits[:, -1, :]

            next_token = combined_sample(
                next_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                generated_ids=generated_ids,
            )

            if next_token.item() in self.stop_tokens:
                break

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            yield token_text

    def generate_with_callback(
        self,
        input_ids: torch.Tensor,
        callback: Callable[[str], None],
        **kwargs: Any,
    ) -> str:
        """Generate text with a callback for each token.

        Args:
            input_ids: Initial input token IDs.
            callback: Function called with each new text chunk.

        Returns:
            Complete generated text.
        """
        full_text = []
        for chunk in self.generate_stream(input_ids, **kwargs):
            callback(chunk)
            full_text.append(chunk)
        return "".join(full_text)

    def generate_full(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> str:
        """Generate complete text (non-streaming).

        Args:
            input_ids: Initial input token IDs.

        Returns:
            Complete generated text.
        """
        chunks = list(self.generate_stream(input_ids, **kwargs))
        return "".join(chunks)
