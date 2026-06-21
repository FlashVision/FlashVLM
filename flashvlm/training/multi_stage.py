"""Multi-stage training for VLMs: alignment pretraining + visual instruction tuning."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from flashvlm.cfg.config import FlashVLMConfig

logger = logging.getLogger(__name__)


class MultiStageTrainer:
    """Multi-stage VLM training pipeline.

    Stage 1 — Alignment pretraining:
        Trains only the vision-language projector while keeping both the
        vision encoder and language model frozen. Uses image-caption pairs
        to learn the mapping between visual and text embedding spaces.

    Stage 2 — Visual instruction tuning:
        Fine-tunes the language model (and optionally the projector) on
        multimodal instruction-following data with the vision encoder frozen.
        Supports full fine-tuning or LoRA-based parameter-efficient training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: FlashVLMConfig,
        output_dir: str = "outputs/multi_stage",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._global_step = 0
        self._current_stage = 0

    def train(
        self,
        stage1_data: Optional[DataLoader] = None,
        stage2_data: Optional[DataLoader] = None,
        val_data: Optional[DataLoader] = None,
        stage1_epochs: int = 1,
        stage2_epochs: int = 3,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 2e-5,
        use_lora_stage2: bool = False,
        lora_rank: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.03,
        save_each_stage: bool = True,
    ) -> Dict[str, Any]:
        """Run the full multi-stage training pipeline.

        Args:
            stage1_data: DataLoader for alignment pretraining (image-caption pairs).
            stage2_data: DataLoader for instruction tuning.
            val_data: Optional validation DataLoader.
            stage1_epochs: Number of epochs for stage 1.
            stage2_epochs: Number of epochs for stage 2.
            stage1_lr: Learning rate for stage 1 projector training.
            stage2_lr: Learning rate for stage 2 fine-tuning.
            use_lora_stage2: Use LoRA for stage 2 instead of full fine-tuning.
            lora_rank: LoRA rank if use_lora_stage2 is True.
            lora_target_modules: Modules to apply LoRA to.
            gradient_accumulation_steps: Gradient accumulation steps.
            max_grad_norm: Maximum gradient norm for clipping.
            warmup_ratio: Warmup ratio of total steps.
            save_each_stage: Save checkpoint after each stage.

        Returns:
            Combined metrics from both stages.
        """
        all_metrics = {}

        logger.info("=" * 60)
        logger.info("STAGE 1: Alignment Pretraining (Projector Only)")
        logger.info("=" * 60)
        self._current_stage = 1

        self._freeze_for_stage1()
        trainable_s1 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_s1:,} / {total:,}")

        if stage1_data is not None:
            s1_metrics = self._train_stage(
                dataloader=stage1_data,
                val_dataloader=val_data,
                epochs=stage1_epochs,
                lr=stage1_lr,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                stage_name="stage1_alignment",
            )
            all_metrics["stage1"] = s1_metrics

            if save_each_stage:
                self._save_checkpoint("stage1_alignment")
        else:
            logger.info("No stage1 data provided, skipping alignment pretraining.")
            all_metrics["stage1"] = {"skipped": True}

        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Visual Instruction Tuning")
        logger.info("=" * 60)
        self._current_stage = 2

        if use_lora_stage2:
            self._apply_lora_stage2(lora_rank, lora_target_modules)
        else:
            self._freeze_for_stage2()

        trainable_s2 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_s2:,} / {total:,}")
        if use_lora_stage2:
            logger.info(f"Using LoRA with rank={lora_rank}")

        if stage2_data is not None:
            s2_metrics = self._train_stage(
                dataloader=stage2_data,
                val_dataloader=val_data,
                epochs=stage2_epochs,
                lr=stage2_lr,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                stage_name="stage2_instruction_tuning",
            )
            all_metrics["stage2"] = s2_metrics

            if save_each_stage:
                self._save_checkpoint("stage2_instruction_tuning")
        else:
            logger.info("No stage2 data provided, skipping instruction tuning.")
            all_metrics["stage2"] = {"skipped": True}

        return all_metrics

    def _freeze_for_stage1(self) -> None:
        """Stage 1: only projector parameters are trainable."""
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(key in name for key in ["projector", "mm_projector", "image_embedding.projector"]):
                param.requires_grad = True

    def _freeze_for_stage2(self) -> None:
        """Stage 2: LLM + projector trainable, vision encoder frozen."""
        for param in self.model.parameters():
            param.requires_grad = True

        for name, param in self.model.named_parameters():
            if any(key in name for key in [
                "vision_encoder", "vision_tower", "frame_encoder",
                "image_embedding.vision_encoder",
            ]):
                param.requires_grad = False

    def _apply_lora_stage2(
        self, rank: int, target_modules: Optional[List[str]],
    ) -> None:
        """Apply LoRA adapters for parameter-efficient stage 2 training."""
        for param in self.model.parameters():
            param.requires_grad = False

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        try:
            from flashvlm.models.lora import apply_lora
            self.model = apply_lora(
                self.model, rank=rank, target_modules=target_modules,
            )
        except ImportError:
            logger.warning("LoRA module not available. Falling back to full fine-tuning.")
            self._freeze_for_stage2()

    def _train_stage(
        self,
        dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        epochs: int,
        lr: float,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        warmup_ratio: float,
        stage_name: str,
    ) -> Dict[str, Any]:
        """Train a single stage."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

        total_steps = len(dataloader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        self.model.train()
        epoch_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                batch = self._move_batch(batch)
                outputs = self._forward_step(batch)
                loss = outputs.get("loss")

                if loss is None:
                    continue

                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self._global_step += 1

                running_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

            avg_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info(f"[{stage_name}] Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                logger.info(f"  val_loss: {val_loss:.4f}")

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "epoch_losses": epoch_losses,
            "total_steps": self._global_step,
        }

    def _forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run a single forward step, handling different batch formats."""
        if "input_ids" in batch:
            return self.model(
                input_ids=batch.get("input_ids"),
                pixel_values=batch.get("pixel_values"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
        if "pixel_values" in batch:
            return self.model(pixel_values=batch["pixel_values"])
        return self.model(**batch)

    def _validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch)
                outputs = self._forward_step(batch)
                loss = outputs.get("loss")
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
        self.model.train()
        return total_loss / max(num_batches, 1)

    def _move_batch(self, batch: Any) -> Dict[str, Any]:
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return {"pixel_values": batch[0].to(self.device)}
        return {"pixel_values": batch.to(self.device)}

    def _save_checkpoint(self, stage_name: str) -> None:
        ckpt_dir = self.output_dir / stage_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        self.config.save_yaml(ckpt_dir / "config.yaml")
        logger.info(f"Saved checkpoint to {ckpt_dir}")
