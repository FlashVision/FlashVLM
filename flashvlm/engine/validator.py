"""Validation engine for FlashVLM models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashvlm.cfg.config import FlashVLMConfig


class Validator:
    """Validation engine that computes metrics during training."""

    def __init__(self, model: nn.Module, config: FlashVLMConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation on a dataloader and return metrics.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of metric names to values.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validating", leave=False):
            batch = self._move_to_device(batch)

            outputs = self.model(**batch)
            loss = outputs.get("loss")
            logits = outputs.get("logits")

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

            if logits is not None and "labels" in batch:
                labels = batch["labels"]
                predictions = logits.argmax(dim=-1)
                mask = labels != -100
                total_correct += (predictions[mask] == labels[mask]).sum().item()
                total_tokens += mask.sum().item()

        self.model.train()

        metrics = {}
        if num_batches > 0:
            metrics["val_loss"] = total_loss / num_batches
        if total_tokens > 0:
            metrics["val_accuracy"] = total_correct / total_tokens
            metrics["val_perplexity"] = torch.exp(
                torch.tensor(total_loss / max(num_batches, 1))
            ).item()

        return metrics

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved
