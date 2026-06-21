# Training Guide

## Training Strategies

FlashVLM supports three training strategies:

1. **Supervised Fine-Tuning (SFT)** — Standard instruction tuning
2. **Direct Preference Optimization (DPO)** — Alignment without reward models
3. **RLHF** — Full reinforcement learning from human feedback

## Supervised Fine-Tuning

### Multi-Stage Training (Recommended)

**Stage 1**: Train only the projector (vision encoder + LLM frozen)
**Stage 2**: Fine-tune the full model (or with LoRA)

```python
from flashvlm import FlashVLM, get_config
from flashvlm.training import SupervisedFineTuner

config = get_config("configs/flashvlm_llava_7b.yaml")
model = FlashVLM(config)

sft = SupervisedFineTuner(model, config)
sft.train_multi_stage(
    train_dataloader=train_loader,
    stage1_epochs=1,
    stage2_epochs=3,
)
```

### LoRA Fine-Tuning

```python
from flashvlm import FlashVLM, apply_lora

model = FlashVLM.from_pretrained("llava-v1.5-7b")
model = apply_lora(model, rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

### CLI Training

```bash
flashvlm train --config configs/flashvlm_llava_7b.yaml --epochs 3 --output-dir outputs/
```

## DPO Training

```python
from flashvlm.training import DPOTrainer

dpo = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    config=config,
    beta=0.1,
)
dpo.train(preference_dataloader, epochs=1)
```

## Data Format

### VQA Dataset (JSON)
```json
[
  {"image": "img001.jpg", "question": "What color is the car?", "answer": "red"},
  {"image": "img002.jpg", "question": "How many people?", "answer": "3"}
]
```

### Captioning Dataset (JSON)
```json
[
  {"image": "img001.jpg", "caption": "A red car parked on a street."},
  {"image": "img002.jpg", "caption": "Three people walking in a park."}
]
```

## Configuration

Key training parameters in YAML configs:

```yaml
training:
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  bf16: true
  deepspeed: null
```

## Tips

- Start with projector-only training to avoid catastrophic forgetting
- Use BF16 mixed precision for faster training with minimal quality loss
- Gradient accumulation helps with limited VRAM
- Monitor validation loss to avoid overfitting
