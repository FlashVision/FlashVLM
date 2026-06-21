"""Configuration system for FlashVLM models and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VisionConfig:
    """Vision encoder configuration."""

    encoder_name: str = "openai/clip-vit-large-patch14-336"
    image_size: int = 336
    patch_size: int = 14
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    freeze: bool = True
    use_checkpoint: bool = False


@dataclass
class ProjectorConfig:
    """Vision-to-language projector configuration."""

    type: str = "mlp"  # mlp, qformer, cross_attention
    input_dim: int = 1024
    output_dim: int = 4096
    num_layers: int = 2
    num_query_tokens: int = 64  # for Q-Former
    dropout: float = 0.0


@dataclass
class LanguageConfig:
    """Language model configuration."""

    model_name: str = "meta-llama/Llama-2-7b-hf"
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
    max_length: int = 2048
    freeze: bool = False
    use_flash_attention: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    use_qlora: bool = False
    bits: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    deepspeed: str | None = None
    output_dir: str = "outputs"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    dataloader_num_workers: int = 4
    seed: int = 42


@dataclass
class GenerationConfig:
    """Text generation configuration."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_data: str | None = None
    val_data: str | None = None
    test_data: str | None = None
    image_dir: str | None = None
    max_length: int = 2048
    image_size: int = 336
    augmentation: bool = True


@dataclass
class FlashVLMConfig:
    """Master configuration for FlashVLM."""

    model_name: str = "flashvlm-7b"
    architecture: str = "llava"  # llava, qwen_vl, internvl, phi_vision
    vision: VisionConfig = field(default_factory=VisionConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a flat dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "__dict__"):
                for sub_key, sub_val in value.__dict__.items():
                    result[f"{key}.{sub_key}"] = sub_val
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlashVLMConfig:
        """Create config from a nested dictionary."""
        config = cls()
        if "vision" in data:
            config.vision = VisionConfig(**data["vision"])
        if "projector" in data:
            config.projector = ProjectorConfig(**data["projector"])
        if "language" in data:
            config.language = LanguageConfig(**data["language"])
        if "lora" in data:
            config.lora = LoRAConfig(**data["lora"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "generation" in data:
            config.generation = GenerationConfig(**data["generation"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "model_name" in data:
            config.model_name = data["model_name"]
        if "architecture" in data:
            config.architecture = data["architecture"]
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> FlashVLMConfig:
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "__dict__"):
                data[key] = {k: v for k, v in value.__dict__.items()}
            else:
                data[key] = value
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_config(path: str | Path | None = None) -> FlashVLMConfig:
    """Get a FlashVLM configuration, optionally loading from a YAML file."""
    if path is None:
        return FlashVLMConfig()
    return FlashVLMConfig.from_yaml(path)
