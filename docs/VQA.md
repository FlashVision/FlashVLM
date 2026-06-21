# Visual Question Answering

## Overview

Visual Question Answering (VQA) is the task of answering natural language questions about images. FlashVLM supports open-ended VQA, multiple-choice, and yes/no questions.

## Basic Usage

```python
from flashvlm import FlashVLM

model = FlashVLM.from_pretrained("llava-v1.5-7b")
answer = model.ask("What is the person doing?", image="photo.jpg")
```

## VQA Task Module

```python
from flashvlm.tasks import VQATask

vqa = VQATask(model, prompt_style="detailed", max_new_tokens=128)
answer = vqa.answer("photo.jpg", "How many cars are parked?")
```

### Prompt Styles

- `default` — "Question: {q}\nAnswer:"
- `llava` — "USER: <image>\n{q}\nASSISTANT:"
- `short` — "{q}\nShort answer:"
- `detailed` — Extended prompt encouraging detailed responses
- `multiple_choice` — Formatted with lettered options

### Multiple Choice

```python
answer = vqa.answer(
    "photo.jpg",
    "What season is depicted?",
    options=["Spring", "Summer", "Fall", "Winter"]
)
```

## Evaluation

Standard VQA accuracy (VQAv2 metric):

```python
predictions = ["red", "3", "yes"]
ground_truths = ["red", "three", "yes"]
metrics = vqa.evaluate(predictions, ground_truths)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## Benchmarks

| Model | VQAv2 | GQA | TextVQA |
|-------|-------|-----|---------|
| LLaVA-1.5-7B | 78.5 | 62.0 | 58.2 |
| Qwen-VL-7B | 78.8 | 59.3 | 63.8 |
| InternVL-7B | 79.3 | 62.9 | 57.0 |

## CLI

```bash
flashvlm vqa --image photo.jpg --question "What color is the car?" --model llava-v1.5-7b
```
