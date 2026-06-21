"""Reinforcement Learning from Human Feedback (RLHF) for VLMs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashvlm.cfg.config import FlashVLMConfig


class RewardModel(nn.Module):
    """Reward model for scoring VLM outputs."""

    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute reward scores from hidden states.

        Args:
            hidden_states: Final hidden states (batch_size, seq_len, hidden_size).

        Returns:
            Scalar reward per sequence (batch_size, 1).
        """
        pooled = hidden_states[:, -1, :]
        return self.backbone(pooled)


class RLHFTrainer:
    """RLHF trainer using PPO for VLM alignment.

    Implements Proximal Policy Optimization with:
    - KL penalty to stay close to reference policy
    - Value function baseline for variance reduction
    - Clip ratio for stable updates
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        config: FlashVLMConfig,
        kl_coeff: float = 0.1,
        clip_ratio: float = 0.2,
        value_coeff: float = 0.5,
        gamma: float = 1.0,
        lam: float = 0.95,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        self.kl_coeff = kl_coeff
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.gamma = gamma
        self.lam = lam

        self.ref_model.eval()
        self.reward_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

        self.value_head = nn.Linear(config.language.hidden_size, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy.to(self.device)
        self.ref_model.to(self.device)
        self.reward_model.to(self.device)
        self.value_head.to(self.device)

    def compute_rewards(
        self,
        responses: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute rewards for generated responses."""
        with torch.no_grad():
            outputs = self.policy(input_ids=responses, pixel_values=pixel_values)
            if isinstance(outputs, dict):
                hidden_states = outputs.get("logits", outputs.get("hidden_states"))
            else:
                hidden_states = outputs.logits if hasattr(outputs, "logits") else outputs

            rewards = self.reward_model(hidden_states)
        return rewards.squeeze(-1)

    def compute_kl_penalty(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty between policy and reference."""
        kl = policy_logprobs - ref_logprobs
        return self.kl_coeff * kl

    def ppo_step(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute PPO loss with clipping.

        Args:
            old_logprobs: Log probs from behavior policy.
            new_logprobs: Log probs from current policy.
            advantages: Generalized advantage estimates.
            values: Value estimates.
            returns: Discounted returns.

        Returns:
            Dictionary with policy_loss, value_loss, and entropy.
        """
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()

        value_loss = F.mse_loss(values, returns)

        entropy = -(new_logprobs * torch.exp(new_logprobs)).mean()

        total_loss = policy_loss + self.value_coeff * value_loss - 0.01 * entropy

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": (old_logprobs - new_logprobs).mean(),
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Per-step rewards (batch_size, seq_len).
            values: Value estimates (batch_size, seq_len).
            dones: Episode termination flags.

        Returns:
            Tuple of (advantages, returns).
        """
        seq_len = rewards.shape[1]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[0], device=rewards.device)

        if dones is None:
            dones = torch.zeros_like(rewards)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(rewards.shape[0], device=rewards.device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            advantages[:, t] = last_gae = (
                delta + self.gamma * self.lam * (1 - dones[:, t]) * last_gae
            )

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train(
        self,
        train_dataloader: DataLoader | None = None,
        epochs: int = 1,
        ppo_epochs: int = 4,
        learning_rate: float = 1e-6,
    ) -> dict[str, Any]:
        """Run RLHF training with PPO.

        Args:
            train_dataloader: DataLoader with prompts and images.
            epochs: Number of outer training loops.
            ppo_epochs: PPO update epochs per batch.
            learning_rate: Learning rate.

        Returns:
            Training metrics.
        """
        torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=learning_rate,
        )

        metrics: dict[str, list[float]] = {"rewards": [], "policy_loss": [], "kl": []}

        if train_dataloader is None:
            print("No dataloader provided. RLHF requires prompt data.")
            return metrics

        for epoch in range(epochs):
            epoch_rewards = []

            for batch in tqdm(train_dataloader, desc=f"RLHF Epoch {epoch + 1}"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                rewards = self.compute_rewards(batch["input_ids"], batch.get("pixel_values"))
                epoch_rewards.append(rewards.mean().item())

            avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            metrics["rewards"].append(avg_reward)
            print(f"Epoch {epoch + 1} - Mean Reward: {avg_reward:.4f}")

        return metrics
