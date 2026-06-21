# Installation

## System Requirements

- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration, optional)
- 16GB RAM minimum (32GB+ recommended for 7B models)
- 20GB+ disk space for model weights

## Quick Install

```bash
pip install flashvlm
```

## From Source (Development)

```bash
git clone https://github.com/FlashVision/FlashVLM.git
cd FlashVLM
pip install -e ".[all]"
```

## Using setup_env.sh

The setup script creates a virtual environment and installs all dependencies:

```bash
bash setup_env.sh
```

## Dependency Groups

### Core (installed by default)
- `torch>=2.1.0` — PyTorch framework
- `torchvision>=0.16.0` — Vision utilities
- `transformers>=4.36.0` — HuggingFace model hub
- `Pillow>=10.0.0` — Image processing
- `numpy>=1.24.0` — Numerical computing
- `PyYAML>=6.0` — Configuration files
- `tqdm>=4.65.0` — Progress bars
- `safetensors>=0.4.0` — Safe model serialization

### Training (optional)
```bash
pip install flashvlm[training]
```
- `peft>=0.7.0` — Parameter-efficient fine-tuning
- `bitsandbytes>=0.41.0` — Quantization (QLoRA)
- `deepspeed>=0.12.0` — Distributed training

### Development (optional)
```bash
pip install flashvlm[dev]
```
- `pytest` — Testing framework
- `ruff` — Code linting
- `pre-commit` — Git hooks

## Docker

```bash
cd docker
docker compose up -d
```

## Verify Installation

```bash
flashvlm check
```

This will verify all dependencies are correctly installed and report GPU availability.

## Troubleshooting

### CUDA not detected
Ensure your NVIDIA drivers and CUDA toolkit are properly installed:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
For large models on limited VRAM, use quantization:
```python
from flashvlm import FlashVLM
model = FlashVLM.from_pretrained("llava-v1.5-7b", load_in_4bit=True)
```

### Import Errors
Ensure you're in the correct virtual environment:
```bash
source .venv/bin/activate
pip install -e .
```
