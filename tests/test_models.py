"""Tests for FlashVLM model components."""

import torch
import torch.nn as nn

from flashvlm.cfg.config import FlashVLMConfig, ProjectorConfig
from flashvlm.models.lora import LoRALinear, apply_lora
from flashvlm.models.projector import CrossAttentionProjector, MLPProjector, QFormerProjector


class TestMLPProjector:
    def test_forward_shape(self):
        config = ProjectorConfig(input_dim=1024, output_dim=4096, num_layers=2)
        proj = MLPProjector(config)

        x = torch.randn(2, 576, 1024)
        output = proj(x)
        assert output.shape == (2, 576, 4096)

    def test_single_layer(self):
        config = ProjectorConfig(input_dim=512, output_dim=768, num_layers=1)
        proj = MLPProjector(config)

        x = torch.randn(1, 100, 512)
        output = proj(x)
        assert output.shape == (1, 100, 768)

    def test_gradient_flow(self):
        config = ProjectorConfig(input_dim=256, output_dim=512, num_layers=2)
        proj = MLPProjector(config)

        x = torch.randn(1, 10, 256, requires_grad=True)
        output = proj(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestQFormerProjector:
    def test_forward_shape(self):
        config = ProjectorConfig(
            input_dim=1024, output_dim=4096, num_layers=2,
            num_query_tokens=64, type="qformer"
        )
        proj = QFormerProjector(config)

        x = torch.randn(2, 576, 1024)
        output = proj(x)
        assert output.shape == (2, 64, 4096)

    def test_different_query_counts(self):
        config = ProjectorConfig(
            input_dim=512, output_dim=768, num_query_tokens=32, type="qformer"
        )
        proj = QFormerProjector(config)

        x = torch.randn(1, 256, 512)
        output = proj(x)
        assert output.shape == (1, 32, 768)


class TestCrossAttentionProjector:
    def test_forward_shape(self):
        config = ProjectorConfig(
            input_dim=1024, output_dim=4096, num_layers=2, type="cross_attention"
        )
        proj = CrossAttentionProjector(config)

        x = torch.randn(2, 576, 1024)
        output = proj(x)
        assert output.shape == (2, 576, 4096)


class TestLoRA:
    def test_lora_linear_forward(self):
        lora = LoRALinear(in_features=768, out_features=768, rank=8, alpha=16)
        x = torch.randn(2, 10, 768)
        output = lora(x)
        assert output.shape == (2, 10, 768)

    def test_lora_from_linear(self):
        linear = nn.Linear(512, 512)
        lora = LoRALinear.from_linear(linear, rank=4, alpha=8)

        x = torch.randn(1, 5, 512)
        output = lora(x)
        assert output.shape == (1, 5, 512)

    def test_lora_merge_weights(self):
        lora = LoRALinear(in_features=64, out_features=64, rank=4)
        x = torch.randn(1, 3, 64)
        out_before = lora(x).detach()
        lora.merge_weights()
        out_after = lora.linear(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_apply_lora_to_model(self):
        model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        _total_before = sum(p.numel() for p in model.parameters())

        class NamedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128)
                self.v_proj = nn.Linear(128, 128)
                self.other = nn.Linear(128, 128)

        model = NamedModel()
        model = apply_lora(model, rank=4, target_modules=["q_proj", "v_proj"])

        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert isinstance(model.other, nn.Linear)

    def test_lora_trainable_params(self):
        lora = LoRALinear(in_features=256, out_features=256, rank=8)
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
        assert trainable < frozen


class TestFlashVLMConfig:
    def test_default_config(self):
        config = FlashVLMConfig()
        assert config.architecture == "llava"
        assert config.vision.image_size == 336
        assert config.language.hidden_size == 4096

    def test_to_dict(self):
        config = FlashVLMConfig()
        d = config.to_dict()
        assert "vision.image_size" in d
        assert d["vision.image_size"] == 336

    def test_from_dict(self):
        data = {
            "model_name": "test-model",
            "architecture": "qwen_vl",
            "vision": {"image_size": 448, "hidden_size": 1024},
        }
        config = FlashVLMConfig.from_dict(data)
        assert config.architecture == "qwen_vl"
        assert config.vision.image_size == 448
