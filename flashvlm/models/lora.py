"""LoRA and QLoRA adapters for efficient fine-tuning."""

from __future__ import annotations

import math
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvlm.cfg.config import LoRAConfig


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer with low-rank weight updates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_output + lora_output * self.scaling

    def merge_weights(self) -> None:
        """Merge LoRA weights into the base linear layer."""
        with torch.no_grad():
            merged = self.lora_B @ self.lora_A * self.scaling
            self.linear.weight.add_(merged)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.05
    ) -> LoRALinear:
        """Create a LoRALinear from an existing nn.Linear layer."""
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        lora_layer.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_layer.linear.bias.data.copy_(linear.bias.data)
        return lora_layer


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to target modules in a model.

    Args:
        model: The base model to adapt.
        rank: LoRA rank (lower = fewer parameters).
        alpha: LoRA scaling factor.
        dropout: Dropout on LoRA inputs.
        target_modules: List of module name patterns to apply LoRA to.
            Defaults to attention projection layers.

    Returns:
        The model with LoRA adapters applied.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    target_set: Set[str] = set(target_modules)
    replaced = 0

    for name, module in list(model.named_modules()):
        for target in target_set:
            if target in name and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)

                lora_layer = LoRALinear.from_linear(
                    module, rank=rank, alpha=alpha, dropout=dropout
                )
                setattr(parent, child_name, lora_layer)
                replaced += 1
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA applied: {replaced} layers adapted, "
        f"{trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def apply_qlora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Apply QLoRA (quantized LoRA) to a model.

    Requires bitsandbytes for 4-bit quantization.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("QLoRA requires bitsandbytes: pip install bitsandbytes")

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            for target in config.target_modules:
                if target in name:
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    parent = model
                    if parent_name:
                        for part in parent_name.split("."):
                            parent = getattr(parent, part)

                    quantized = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.bfloat16,
                        quant_type="nf4",
                    )
                    quantized.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        quantized.bias.data.copy_(module.bias.data)

                    setattr(parent, child_name, quantized)
                    break

    model = apply_lora(
        model,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=config.target_modules,
    )
    return model
