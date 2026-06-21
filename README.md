<div align="center">

# FlashVLM

**High-Performance Vision-Language Models for Multimodal AI**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg)](https://huggingface.co/FlashVision)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

*Unified framework for training, fine-tuning, and deploying Vision-Language Models*

[Installation](#installation) | [Quick Start](#quick-start) | [Models](#supported-models) | [Documentation](docs/) | [Examples](examples/)

</div>

---

## Overview

FlashVLM is a production-ready framework for Vision-Language Models (VLMs) that unifies leading architectures under a single, consistent API. It supports visual question answering, image captioning, visual grounding, document understanding, and multimodal reasoning.

### Key Features

- **Unified Architecture Support** — LLaVA, Qwen-VL, InternVL, and Phi-Vision under one API
- **Multiple Vision Encoders** — CLIP, SigLIP, DINOv2 with flexible projection layers
- **Efficient Training** — LoRA/QLoRA, DeepSpeed ZeRO, gradient checkpointing
- **Advanced Generation** — Streaming, beam search, top-k/top-p sampling
- **Task Modules** — VQA, captioning, grounding, OCR, chain-of-thought reasoning
- **Production Solutions** — Multimodal chatbot, document QA, image analysis
- **RLHF & DPO** — Alignment training for safer, more helpful VLMs

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/FlashVision/FlashVLM.git
cd FlashVLM
pip install -e ".[all]"
```

### Quick Install

```bash
pip install flashvlm
```

### With Training Dependencies

```bash
pip install flashvlm[training]
```

### Environment Setup

```bash
bash setup_env.sh
```

## Quick Start

### Visual Question Answering

```python
from flashvlm import FlashVLM

model = FlashVLM.from_pretrained("llava-v1.5-7b")
answer = model.ask("What is happening in this image?", image="photo.jpg")
print(answer)
```

### Image Captioning

```python
from flashvlm import FlashVLM

model = FlashVLM.from_pretrained("llava-v1.5-7b")
caption = model.caption("photo.jpg")
print(caption)
```

### Multimodal Chat

```python
from flashvlm.solutions import MultimodalChat

chat = MultimodalChat(model_name="llava-v1.5-7b")
chat.add_image("diagram.png")
response = chat.send("Explain this diagram step by step.")
print(response)
```

### CLI Usage

```bash
# Visual QA
flashvlm vqa --image photo.jpg --question "What color is the car?"

# Image captioning
flashvlm caption --image photo.jpg --max-tokens 100

# Interactive chat
flashvlm chat --model llava-v1.5-7b

# Export model
flashvlm export --model llava-v1.5-7b --format onnx

# Run benchmarks
flashvlm benchmark --model llava-v1.5-7b --dataset vqav2
```

## Supported Models

| Model | Parameters | Vision Encoder | Architecture | Tasks |
|-------|-----------|----------------|--------------|-------|
| LLaVA-1.5-7B | 7B | CLIP ViT-L/14 | LLaVA | VQA, Caption, Chat |
| Qwen-VL-7B | 7B | ViT-bigG | Qwen-VL | VQA, OCR, Grounding |
| InternVL-7B | 7B | InternViT-6B | InternVL | VQA, Caption, Reasoning |
| Phi-Vision-4B | 4B | SigLIP | Phi-Vision | VQA, Caption, Chat |

## Training

### Supervised Fine-Tuning

```python
from flashvlm import FlashVLM, Trainer
from flashvlm.cfg import get_config

config = get_config("configs/flashvlm_llava_7b.yaml")
model = FlashVLM(config)
trainer = Trainer(model, config)
trainer.train()
```

### LoRA Fine-Tuning

```python
from flashvlm import FlashVLM, apply_lora

model = FlashVLM.from_pretrained("llava-v1.5-7b")
model = apply_lora(model, rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

## Project Structure

```
FlashVLM/
├── configs/          # Model configuration YAML files
├── docker/           # Docker deployment files
├── docs/             # Documentation
├── examples/         # Usage examples
├── flashvlm/         # Main package
│   ├── cfg/          # Configuration management
│   ├── data/         # Datasets and transforms
│   ├── engine/       # Training and inference engines
│   ├── models/       # VLM architectures
│   ├── generation/   # Text generation utilities
│   ├── tasks/        # Task-specific modules
│   ├── training/     # Training strategies (SFT, DPO, RLHF)
│   ├── solutions/    # End-to-end solutions
│   ├── analytics/    # Benchmarking and metrics
│   └── utils/        # Utility functions
├── tests/            # Unit tests
├── pyproject.toml    # Package configuration
└── README.md         # This file
```

## Benchmarks

| Model | VQAv2 | GQA | TextVQA | MM-Bench |
|-------|-------|-----|---------|----------|
| LLaVA-1.5-7B | 78.5 | 62.0 | 58.2 | 64.3 |
| Qwen-VL-7B | 78.8 | 59.3 | 63.8 | 38.2 |
| InternVL-7B | 79.3 | 62.9 | 57.0 | 65.4 |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{flashvlm2024,
  title={FlashVLM: High-Performance Vision-Language Models},
  author={FlashVision Team},
  year={2024},
  url={https://github.com/FlashVision/FlashVLM}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
