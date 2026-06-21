"""Direct Preference Optimization (DPO) for VLMs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashvlm.cfg.config import FlashVLMConfig


class DPOTrainer:
    """Direct Preference Optimization trainer for Vision-Language Models.

    Implements DPO loss to align VLM outputs with human preferences
    without explicit reward modeling.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: FlashVLMConfig,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.beta = beta
        self.label_smoothing = label_smoothing

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ref_model.to(self.device)

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen responses under policy.
            policy_rejected_logps: Log probs of rejected responses under policy.
            reference_chosen_logps: Log probs of chosen responses under reference.
            reference_rejected_logps: Log probs of rejected responses under reference.

        Returns:
            Tuple of (loss, chosen_reward, rejected_reward).
        """
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        logits = self.beta * (chosen_logratios - rejected_logratios)

        if self.label_smoothing > 0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            loss = -F.logsigmoid(logits)

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def get_batch_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a batch under a given model."""
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        labels_shifted = labels[:, 1:]
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=labels_shifted.unsqueeze(-1)
        ).squeeze(-1)

        mask = (labels_shifted != -100).float()
        sequence_logps = (per_token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return sequence_logps

    def train(
        self,
        train_dataloader: DataLoader | None = None,
        epochs: int = 1,
        learning_rate: float = 5e-7,
    ) -> dict[str, Any]:
        """Run DPO training.

        Expects dataloader batches with:
        - chosen_input_ids, chosen_attention_mask, chosen_labels
        - rejected_input_ids, rejected_attention_mask, rejected_labels
        - pixel_values (optional)

        Args:
            train_dataloader: DataLoader with preference pairs.
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimization.

        Returns:
            Training metrics.
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        self.model.train()
        all_losses = []

        if train_dataloader is None:
            print("No dataloader provided. DPO training requires preference data.")
            return {"losses": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_dataloader, desc=f"DPO Epoch {epoch + 1}"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                pixel_values = batch.get("pixel_values")

                policy_chosen_logps = self.get_batch_logps(
                    self.model,
                    batch["chosen_input_ids"],
                    pixel_values,
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                policy_rejected_logps = self.get_batch_logps(
                    self.model,
                    batch["rejected_input_ids"],
                    pixel_values,
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

                with torch.no_grad():
                    ref_chosen_logps = self.get_batch_logps(
                        self.ref_model,
                        batch["chosen_input_ids"],
                        pixel_values,
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                    )
                    ref_rejected_logps = self.get_batch_logps(
                        self.ref_model,
                        batch["rejected_input_ids"],
                        pixel_values,
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                    )

                loss, chosen_reward, rejected_reward = self.compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            all_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} - DPO Loss: {avg_loss:.4f}")

        return {"losses": all_losses}
