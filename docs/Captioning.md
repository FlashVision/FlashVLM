# Image Captioning

## Overview

Image captioning generates natural language descriptions of images. FlashVLM supports multiple captioning styles from brief one-line descriptions to detailed paragraphs.

## Basic Usage

```python
from flashvlm import FlashVLM

model = FlashVLM.from_pretrained("llava-v1.5-7b")
caption = model.caption("photo.jpg")
```

## Captioning Task Module

```python
from flashvlm.tasks import CaptioningTask

captioner = CaptioningTask(model, style="detailed", max_new_tokens=200)
caption = captioner.caption("photo.jpg")
```

### Caption Styles

| Style | Description | Example Output |
|-------|-------------|----------------|
| `brief` | One sentence | "A cat on a couch." |
| `detailed` | Full description | "A tabby cat lounging on a blue velvet couch..." |
| `creative` | Engaging narrative | "In the afternoon sun, a content feline..." |
| `technical` | Structured analysis | "Subject: domestic cat (tabby). Position: center-right..." |
| `accessibility` | Screen reader friendly | "A medium-sized tabby cat with orange stripes..." |

### Contextual Captioning

```python
caption = captioner.caption_with_context(
    "xray.jpg",
    context="This is a chest X-ray from a hospital"
)
```

## Batch Processing

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
captions = captioner.batch_caption(images, style="brief")
```

## Evaluation Metrics

FlashVLM implements standard captioning metrics:

- **BLEU-4** — N-gram precision
- **CIDEr** — Consensus-based scoring
- **METEOR** — Word matching with synonyms

```python
from flashvlm.analytics.metrics import compute_bleu, compute_cider

bleu = compute_bleu(predictions, references)
cider = compute_cider(predictions, references)
```

## CLI

```bash
flashvlm caption --image photo.jpg --detail detailed --max-tokens 200
```
