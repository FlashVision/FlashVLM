# Supported Models

FlashVLM supports multiple VLM architectures through a unified interface.

## Architecture Overview

All VLMs in FlashVLM follow a three-component design:

1. **Vision Encoder** — Extracts visual features from images
2. **Projector** — Bridges vision and language representations
3. **Language Model** — Generates text responses

## LLaVA (Large Language and Vision Assistant)

**Architecture**: CLIP ViT-L/14 → 2-layer MLP → LLaMA-2 7B

```python
from flashvlm import FlashVLM
model = FlashVLM.from_pretrained("llava-v1.5-7b")
```

- Simple and effective MLP projection
- Strong general-purpose performance
- Well-suited for VQA and captioning

## Qwen-VL

**Architecture**: ViT-bigG → Cross-Attention Resampler → Qwen-7B

```python
model = FlashVLM.from_pretrained("qwen-vl-7b")
```

- Visual tokenizer with spatial position encoding
- Strong OCR and grounding capabilities
- Supports interleaved image-text input

## InternVL

**Architecture**: InternViT-6B → Pixel Shuffle + MLP → InternLM2-7B

```python
model = FlashVLM.from_pretrained("internvl-7b")
```

- Dynamic resolution with tile-based processing
- Pixel shuffle for efficient token reduction
- Best for complex visual reasoning

## Phi-Vision

**Architecture**: SigLIP → MLP → Phi-2

```python
model = FlashVLM.from_pretrained("phi-vision-4b")
```

- Compact and efficient (4B parameters)
- SigLIP vision encoder for better representations
- Ideal for resource-constrained environments

## Vision Encoders

| Encoder | Resolution | Patches | Hidden Size |
|---------|-----------|---------|-------------|
| CLIP ViT-L/14 | 336×336 | 576 | 1024 |
| SigLIP-SO400M | 384×384 | 729 | 1152 |
| DINOv2-L | 518×518 | 1369 | 1024 |
| InternViT-6B | 448×448 | 1024 | 3200 |

## Projector Types

| Type | Mechanism | Output Tokens | Use Case |
|------|-----------|---------------|----------|
| MLP | Linear projection | Same as input | LLaVA, InternVL |
| Q-Former | Cross-attention | Fixed (64-256) | Qwen-VL, BLIP-2 |
| Cross-Attention | Multi-layer attention | Same as input | Research |

## Custom Models

Create custom architectures by combining components:

```python
from flashvlm.cfg import FlashVLMConfig
from flashvlm import FlashVLM

config = FlashVLMConfig()
config.architecture = "llava"
config.vision.encoder_name = "openai/clip-vit-large-patch14-336"
config.projector.type = "mlp"
config.language.model_name = "meta-llama/Llama-2-7b-hf"

model = FlashVLM(config)
```
