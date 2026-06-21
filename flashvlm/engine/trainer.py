"""Training engine for FlashVLM models."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashvlm.cfg.config import FlashVLMConfig


class Trainer:
    """Training engine for FlashVLM models with mixed precision and gradient accumulation."""

    def __init__(self, model: nn.Module, config: FlashVLMConfig):
        self.model = model
        self.config = config
        self.train_cfg = config.training

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda") if self.train_cfg.fp16 else None

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.train_losses: list[float] = []

    def _build_optimizer(self) -> AdamW:
        """Build optimizer with weight decay applied to non-bias/norm parameters."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=self.train_cfg.learning_rate)

    def _build_scheduler(self, num_training_steps: int):
        """Build learning rate scheduler with warmup."""
        warmup_steps = int(num_training_steps * self.train_cfg.warmup_ratio)

        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        decay = CosineAnnealingLR(self.optimizer, T_max=num_training_steps - warmup_steps)

        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )

    def train(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Run the training loop.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            callbacks: Optional list of callback objects.

        Returns:
            Dictionary of training metrics.
        """
        if train_dataloader is None:
            print("No training dataloader provided. Creating a synthetic one for demonstration.")
            train_dataloader = self._create_synthetic_dataloader()

        num_training_steps = (
            len(train_dataloader) // self.train_cfg.gradient_accumulation_steps
        ) * self.train_cfg.epochs
        self._build_scheduler(num_training_steps)

        output_dir = Path(self.train_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        print(f"Training for {self.train_cfg.epochs} epochs, {num_training_steps} total steps")
        print(f"  Batch size: {self.train_cfg.batch_size}")
        print(f"  Gradient accumulation: {self.train_cfg.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.train_cfg.learning_rate}")
        print(f"  Device: {self.device}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.train_cfg.epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(train_dataloader)
            self.train_losses.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{self.train_cfg.epochs} - Loss: {epoch_loss:.4f}")

            if val_dataloader is not None:
                val_metrics = self._validate(val_dataloader)
                print(f"  Validation: {val_metrics}")

            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(output_dir / f"checkpoint-epoch-{epoch + 1}")

        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.1f}s")

        self._save_checkpoint(output_dir / "final")
        return {"train_losses": self.train_losses, "total_steps": self.global_step}

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):
            batch = self._move_to_device(batch)

            use_amp = self.train_cfg.fp16 or self.train_cfg.bf16
            dtype = torch.bfloat16 if self.train_cfg.bf16 else torch.float16

            if use_amp and self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=dtype):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    if loss is not None:
                        loss = loss / self.train_cfg.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"]
                if loss is not None:
                    loss = loss / self.train_cfg.gradient_accumulation_steps

            if loss is not None:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item() * self.train_cfg.gradient_accumulation_steps
                num_batches += 1

            if (step + 1) % self.train_cfg.gradient_accumulation_steps == 0:
                if self.train_cfg.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.max_grad_norm
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.train_cfg.logging_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        return total_loss / max(num_batches, 1)

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        from flashvlm.engine.validator import Validator

        validator = Validator(self.model, self.config)
        return validator.validate(dataloader)

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the training device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _save_checkpoint(self, path: Path) -> None:
        """Save a training checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, path / "checkpoint.pt")
        self.config.save_yaml(path / "config.yaml")

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        path = Path(checkpoint_path)
        checkpoint = torch.load(path / "checkpoint.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _create_synthetic_dataloader(self) -> DataLoader:
        """Create a synthetic dataloader for testing the training loop."""
        from torch.utils.data import TensorDataset

        batch_size = self.train_cfg.batch_size
        seq_len = 128
        vocab_size = self.config.language.vocab_size
        img_size = self.config.vision.image_size

        num_samples = batch_size * 10
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        pixel_values = torch.randn(num_samples, 3, img_size, img_size)
        attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
        labels = torch.randint(0, vocab_size, (num_samples, seq_len))

        dataset = TensorDataset(input_ids, pixel_values, attention_mask, labels)

        def collate_fn(batch):
            input_ids, pixels, masks, labels = zip(*batch)
            return {
                "input_ids": torch.stack(input_ids),
                "pixel_values": torch.stack(pixels),
                "attention_mask": torch.stack(masks),
                "labels": torch.stack(labels),
            }

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
