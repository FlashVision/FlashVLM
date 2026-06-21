# Frequently Asked Questions

## General

### What models does FlashVLM support?

FlashVLM supports LLaVA-1.5, Qwen-VL, InternVL, and Phi-Vision architectures. Each can be loaded with a single line of code.

### Do I need a GPU?

A GPU is strongly recommended for inference (especially with 7B models). CPU inference is supported but significantly slower. The Phi-Vision-4B model is the most efficient option for limited hardware.

### How much VRAM do I need?

- Phi-Vision-4B: ~8GB VRAM
- LLaVA-7B (FP16): ~14GB VRAM
- LLaVA-7B (4-bit): ~6GB VRAM
- InternVL-7B: ~20GB VRAM

### Can I use FlashVLM without internet?

Yes, once model weights are downloaded. Use `from_pretrained()` with a local path.

## Training

### How do I fine-tune on my own data?

Prepare your data in JSON format and use the training configs:

```python
from flashvlm import FlashVLM, Trainer, get_config

config = get_config("configs/flashvlm_llava_7b.yaml")
config.data.train_data = "path/to/your/data.json"
config.data.image_dir = "path/to/images/"

model = FlashVLM(config)
trainer = Trainer(model, config)
trainer.train()
```

### What is LoRA and when should I use it?

LoRA (Low-Rank Adaptation) adds small trainable matrices to existing layers, reducing trainable parameters by 90%+. Use it when:
- You have limited VRAM
- You want to preserve base model capabilities
- You're fine-tuning on a specific task

### How long does training take?

Depends on dataset size, hardware, and configuration:
- 10K samples, LoRA, single A100: ~1 hour
- 100K samples, full fine-tune, 4×A100: ~8 hours
- 1M samples, multi-stage: ~2-3 days

## Inference

### How do I speed up inference?

1. Use FP16 or BF16 precision
2. Enable Flash Attention
3. Use 4-bit quantization for limited VRAM
4. Batch multiple inputs together

### Can I stream model output?

Yes, use the StreamingGenerator:

```python
from flashvlm.generation import StreamingGenerator

generator = StreamingGenerator(model, tokenizer)
for token in generator.generate_stream(input_ids):
    print(token, end="", flush=True)
```

### How do I export for production?

```bash
flashvlm export --model llava-v1.5-7b --format onnx --output deployed_model/
```

## Troubleshooting

### "CUDA out of memory"

- Reduce batch size
- Use gradient accumulation
- Enable 4-bit quantization
- Use LoRA instead of full fine-tuning

### Model produces repetitive text

- Increase `repetition_penalty` (try 1.2-1.5)
- Lower `temperature` (try 0.3-0.7)
- Use `top_p=0.9` with `top_k=50`

### Image not being processed correctly

Ensure images are in RGB format and accessible:
```python
from PIL import Image
img = Image.open("path.jpg").convert("RGB")
```
