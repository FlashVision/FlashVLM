"""Training callbacks for FlashVLM."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class TrainingCallback:
    """Base callback class for training hooks."""

    def on_train_begin(self, **kwargs: Any) -> None:
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        pass

    def on_step_begin(self, step: int, **kwargs: Any) -> None:
        pass

    def on_step_end(self, step: int, loss: float, **kwargs: Any) -> None:
        pass

    def on_evaluate(self, metrics: dict[str, float], **kwargs: Any) -> None:
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on validation metric.

    Stops training when the monitored metric hasn't improved
    for a specified number of epochs.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return

        if self.best_value is None:
            self.best_value = value
            return

        improved = False
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        elif self.mode == "max":
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"Early stopping triggered. {self.monitor} hasn't improved "
                    f"for {self.patience} epochs. Best: {self.best_value:.4f}"
                )


class LoggingCallback(TrainingCallback):
    """Logging callback that writes metrics to file."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training.log"
        self.start_time: float | None = None

    def on_train_begin(self, **kwargs: Any) -> None:
        self.start_time = time.time()
        with open(self.log_file, "a") as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        elapsed = time.time() - (self.start_time or time.time())
        with open(self.log_file, "a") as f:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            f.write(f"Epoch {epoch} [{elapsed:.1f}s]: {metrics_str}\n")

    def on_train_end(self, **kwargs: Any) -> None:
        elapsed = time.time() - (self.start_time or time.time())
        with open(self.log_file, "a") as f:
            f.write(f"Training completed in {elapsed:.1f}s\n")


class CheckpointCallback(TrainingCallback):
    """Save model checkpoints at regular intervals."""

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_every_n_epochs: int = 1,
        keep_last_n: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.saved_checkpoints: list[Path] = []

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        if (epoch + 1) % self.save_every_n_epochs != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        import torch

        ckpt_path = self.save_dir / f"checkpoint-epoch-{epoch + 1}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        self.saved_checkpoints.append(ckpt_path)

        while len(self.saved_checkpoints) > self.keep_last_n:
            old_ckpt = self.saved_checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
