"""Sampling strategies: top-k, top-p (nucleus), temperature scaling."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


class TemperatureSampler:
    """Apply temperature scaling to logits before sampling."""

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.

        Args:
            logits: Raw logits of shape (batch_size, vocab_size).

        Returns:
            Temperature-scaled logits.
        """
        return logits / self.temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token IDs from temperature-scaled distribution."""
        scaled = self(logits)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopKSampler:
    """Top-k sampling: restrict to the k most probable tokens."""

    def __init__(self, k: int = 50, temperature: float = 1.0):
        self.k = k
        self.temperature = TemperatureSampler(temperature)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k filtering to logits.

        Args:
            logits: Raw logits of shape (batch_size, vocab_size).

        Returns:
            Filtered logits with non-top-k values set to -inf.
        """
        logits = self.temperature(logits)
        top_k_values, _ = torch.topk(logits, self.k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        logits[logits < threshold] = float("-inf")
        return logits

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token IDs using top-k filtering."""
        filtered = self(logits)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopPSampler:
    """Top-p (nucleus) sampling: restrict to tokens whose cumulative probability >= p."""

    def __init__(self, p: float = 0.9, temperature: float = 1.0, min_tokens_to_keep: int = 1):
        if not 0.0 < p <= 1.0:
            raise ValueError("p must be in (0, 1].")
        self.p = p
        self.min_tokens_to_keep = min_tokens_to_keep
        self.temperature = TemperatureSampler(temperature)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply nucleus (top-p) filtering.

        Args:
            logits: Raw logits of shape (batch_size, vocab_size).

        Returns:
            Filtered logits.
        """
        logits = self.temperature(logits)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= self.p
        if self.min_tokens_to_keep > 0:
            sorted_mask[:, : self.min_tokens_to_keep] = False

        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits[indices_to_remove] = float("-inf")
        return logits

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token IDs using nucleus filtering."""
        filtered = self(logits)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class RepetitionPenaltySampler:
    """Apply repetition penalty to reduce token repetition."""

    def __init__(self, penalty: float = 1.2):
        self.penalty = penalty

    def __call__(self, logits: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
        """Apply repetition penalty based on previously generated tokens.

        Args:
            logits: Current step logits (batch_size, vocab_size).
            generated_ids: Previously generated token IDs (batch_size, seq_len).

        Returns:
            Penalized logits.
        """
        for batch_idx in range(logits.shape[0]):
            unique_tokens = generated_ids[batch_idx].unique()
            for token_id in unique_tokens:
                if token_id < 0:
                    continue
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= self.penalty
                else:
                    logits[batch_idx, token_id] *= self.penalty
        return logits


def combined_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply combined sampling strategy.

    Args:
        logits: Raw logits (batch_size, vocab_size).
        temperature: Temperature for scaling.
        top_k: Top-k value (0 to disable).
        top_p: Top-p value (1.0 to disable).
        repetition_penalty: Repetition penalty (1.0 to disable).
        generated_ids: Previously generated tokens for repetition penalty.

    Returns:
        Sampled token IDs (batch_size,).
    """
    if repetition_penalty != 1.0 and generated_ids is not None:
        rep_sampler = RepetitionPenaltySampler(repetition_penalty)
        logits = rep_sampler(logits, generated_ids)

    logits = logits / temperature

    if top_k > 0:
        top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        logits[logits < threshold] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
        mask[:, 0] = False
        indices_to_remove = mask.scatter(1, sorted_indices, mask)
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
