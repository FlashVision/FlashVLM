"""Supervised Fine-Tuning (SFT) for VLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flashvlm.cfg.config import FlashVLMConfig
from flashvlm.engine.trainer import Trainer


class SupervisedFineTuner:
    """Supervised fine-tuning with multi-stage training support.

    Stage 1: Projector-only training (vision encoder + LLM frozen)
    Stage 2: Full fine-tuning or LoRA-based fine-tuning
    """

    def __init__(
        self,
        model: nn.Module,
        config: FlashVLMConfig,
        stage: int = 1,
    ):
        self.model = model
        self.config = config
        self.stage = stage
        self._setup_frozen_params()

    def _setup_frozen_params(self) -> None:
        """Configure which parameters are trainable based on stage."""
        if self.stage == 1:
            for name, param in self.model.named_parameters():
                if "projector" in name or "mm_projector" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Stage 1: Training projector only ({trainable:,} / {total:,} params)")

        elif self.stage == 2:
            for name, param in self.model.named_parameters():
                if "vision_encoder" in name or "vision_tower" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Stage 2: Full fine-tuning ({trainable:,} / {total:,} params)")

    def train(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run supervised fine-tuning.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.

        Returns:
            Training metrics dictionary.
        """
        trainer = Trainer(self.model, self.config)
        return trainer.train(train_dataloader, val_dataloader)

    def train_multi_stage(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        stage1_epochs: int = 1,
        stage2_epochs: int = 3,
    ) -> Dict[str, Any]:
        """Run multi-stage training: projector then full model.

        Args:
            train_dataloader: Training data.
            val_dataloader: Validation data.
            stage1_epochs: Epochs for projector warmup.
            stage2_epochs: Epochs for full fine-tuning.

        Returns:
            Combined metrics from both stages.
        """
        print("=" * 50)
        print("Stage 1: Projector Warmup")
        print("=" * 50)
        self.stage = 1
        self._setup_frozen_params()
        self.config.training.epochs = stage1_epochs
        self.config.training.learning_rate = 1e-3
        stage1_metrics = self.train(train_dataloader, val_dataloader)

        print("\n" + "=" * 50)
        print("Stage 2: Full Fine-Tuning")
        print("=" * 50)
        self.stage = 2
        self._setup_frozen_params()
        self.config.training.epochs = stage2_epochs
        self.config.training.learning_rate = 2e-5
        stage2_metrics = self.train(train_dataloader, val_dataloader)

        return {"stage1": stage1_metrics, "stage2": stage2_metrics}
