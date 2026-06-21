# FlashVLM Documentation

Welcome to FlashVLM — a high-performance framework for Vision-Language Models.

## What is FlashVLM?

FlashVLM is a unified Python framework for building, training, and deploying Vision-Language Models (VLMs). It supports state-of-the-art architectures including LLaVA, Qwen-VL, InternVL, and Phi-Vision under a single, consistent API.

## Key Capabilities

- **Visual Question Answering** — Answer questions about images with high accuracy
- **Image Captioning** — Generate detailed, contextual descriptions of images
- **Visual Grounding** — Locate objects in images from natural language descriptions
- **Document Understanding** — Extract and analyze text from documents
- **Visual Reasoning** — Chain-of-thought reasoning about visual content
- **Multimodal Chat** — Interactive conversations about images

## Architecture

FlashVLM follows a modular design:

```
Image → Vision Encoder → Projector → Language Model → Text Output
```

- **Vision Encoder**: CLIP, SigLIP, or DINOv2 for visual feature extraction
- **Projector**: MLP, Q-Former, or Cross-Attention for modality bridging
- **Language Model**: LLaMA, Qwen, InternLM, or Phi for text generation

## Getting Started

1. [Installation](Installation.md) — Set up your environment
2. [Quick Start](Quick-Start.md) — Run your first VLM inference
3. [Models](Models.md) — Explore supported architectures
4. [Training](Training.md) — Fine-tune models on your data

## Navigation

| Page | Description |
|------|-------------|
| [Installation](Installation.md) | Environment setup and dependencies |
| [Quick Start](Quick-Start.md) | First steps with FlashVLM |
| [Models](Models.md) | Supported model architectures |
| [Training](Training.md) | Fine-tuning and training guide |
| [VQA](VQA.md) | Visual Question Answering guide |
| [Captioning](Captioning.md) | Image Captioning guide |
| [FAQ](FAQ.md) | Frequently Asked Questions |
