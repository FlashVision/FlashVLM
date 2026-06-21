"""Vision-to-Language projection modules: MLP, Q-Former, Cross-Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from flashvlm.cfg.config import ProjectorConfig
from flashvlm.registry import PROJECTORS


class Projector(nn.Module):
    """Base projector class."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@PROJECTORS.register("mlp")
class MLPProjector(Projector):
    """Multi-layer perceptron projector (LLaVA-style)."""

    def __init__(self, config: ProjectorConfig):
        super().__init__(config)
        layers = []
        in_dim = config.input_dim
        hidden_dim = config.output_dim

        for i in range(config.num_layers):
            out_dim = config.output_dim if i == config.num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < config.num_layers - 1:
                layers.append(nn.GELU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim

        self.proj = nn.Sequential(*layers)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_features)


@PROJECTORS.register("qformer")
class QFormerProjector(Projector):
    """Q-Former style projector with learnable query tokens and cross-attention."""

    def __init__(self, config: ProjectorConfig):
        super().__init__(config)
        self.num_queries = config.num_query_tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, config.output_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )
        self.kv_proj = nn.Linear(config.input_dim, config.output_dim)
        self.layer_norm_q = nn.LayerNorm(config.output_dim)
        self.layer_norm_kv = nn.LayerNorm(config.output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim * 4, config.output_dim),
        )
        self.layer_norm_ffn = nn.LayerNorm(config.output_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size = visual_features.shape[0]
        queries = self.query_tokens.expand(batch_size, -1, -1)
        kv = self.kv_proj(visual_features)

        queries = self.layer_norm_q(queries)
        kv = self.layer_norm_kv(kv)
        attn_output, _ = self.cross_attn(queries, kv, kv)
        queries = queries + attn_output

        ffn_output = self.ffn(self.layer_norm_ffn(queries))
        output = queries + ffn_output
        return output


@PROJECTORS.register("cross_attention")
class CrossAttentionProjector(Projector):
    """Multi-layer cross-attention projector with gated residuals."""

    def __init__(self, config: ProjectorConfig):
        super().__init__(config)
        self.input_proj = nn.Linear(config.input_dim, config.output_dim)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(config.output_dim, num_heads=8, dropout=config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.output_proj = nn.Linear(config.output_dim, config.output_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(visual_features)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with layer normalization."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + torch.tanh(self.gate) * attn_out

        residual = x
        x = self.layer_norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


def build_projector(config: ProjectorConfig) -> Projector:
    """Factory function to create a projector from config."""
    return PROJECTORS.build(config.type, config)
