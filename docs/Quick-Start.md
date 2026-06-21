# Quick Start

Get up and running with FlashVLM in minutes.

## Basic Usage

### Load a Model

```python
from flashvlm import FlashVLM

model = FlashVLM.from_pretrained("llava-v1.5-7b")
```

### Visual Question Answering

```python
answer = model.ask("What is in this image?", image="photo.jpg")
print(answer)
```

### Image Captioning

```python
caption = model.caption("photo.jpg")
print(caption)
```

### Multi-turn Chat

```python
from flashvlm.solutions import MultimodalChat

chat = MultimodalChat(model_name="llava-v1.5-7b")
chat.add_image("diagram.png")

response = chat.send("What does this diagram show?")
print(response)

response = chat.send("Can you explain step 3 in more detail?")
print(response)
```

## CLI Quick Start

```bash
# Check installation
flashvlm check

# Ask a question about an image
flashvlm vqa --image photo.jpg --question "What color is the car?"

# Generate a caption
flashvlm caption --image photo.jpg --detail detailed

# Start interactive chat
flashvlm chat --model llava-v1.5-7b
```

## Choosing a Model

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| Phi-Vision-4B | Fast | Good | 8GB | Quick inference, edge |
| LLaVA-1.5-7B | Medium | Great | 16GB | General-purpose VQA |
| Qwen-VL-7B | Medium | Great | 16GB | OCR, grounding |
| InternVL-7B | Slower | Best | 24GB | Complex reasoning |

## Configuration

Use YAML configs for reproducible experiments:

```python
from flashvlm import FlashVLM, get_config

config = get_config("configs/flashvlm_llava_7b.yaml")
model = FlashVLM(config)
```

## Next Steps

- [Models](Models.md) — Learn about supported architectures
- [Training](Training.md) — Fine-tune on your data
- [VQA](VQA.md) — Deep dive into Visual QA
- [Captioning](Captioning.md) — Advanced captioning options
