"""VQA, Captioning, and Visual Grounding datasets for FlashVLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from flashvlm.data.transforms import build_transform
from flashvlm.registry import DATASETS


@DATASETS.register("vqa")
class VQADataset(Dataset):
    """Visual Question Answering dataset.

    Supports VQAv2, GQA, TextVQA, and custom VQA JSON formats.
    Expected JSON format: [{"image": "path", "question": "text", "answer": "text"}, ...]
    """

    def __init__(
        self,
        data_path: str | Path,
        image_dir: str | Path,
        tokenizer: Any,
        transform: Callable | None = None,
        max_length: int = 512,
        prompt_template: str = "Question: {question}\nAnswer:",
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform or build_transform(image_size=336)
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.samples = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        """Load dataset from JSON file."""
        if not self.data_path.exists():
            return []
        with open(self.data_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image_path = self.image_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        question = sample["question"]
        answer = sample.get("answer", "")
        prompt = self.prompt_template.format(question=question)
        full_text = f"{prompt} {answer}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        prompt_encoding = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@DATASETS.register("captioning")
class CaptioningDataset(Dataset):
    """Image Captioning dataset.

    Supports COCO-style captions and custom formats.
    Expected JSON format: [{"image": "path", "caption": "text"}, ...]
    """

    def __init__(
        self,
        data_path: str | Path,
        image_dir: str | Path,
        tokenizer: Any,
        transform: Callable | None = None,
        max_length: int = 256,
        prompt: str = "Describe this image in detail.",
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform or build_transform(image_size=336)
        self.max_length = max_length
        self.prompt = prompt
        self.samples = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if not self.data_path.exists():
            return []
        with open(self.data_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "annotations" in data:
                annotations = data["annotations"]
                images = {img["id"]: img["file_name"] for img in data.get("images", [])}
                samples = []
                for ann in annotations:
                    samples.append(
                        {
                            "image": images.get(ann["image_id"], ""),
                            "caption": ann["caption"],
                        }
                    )
                return samples
            elif "data" in data:
                return data["data"]
        return data if isinstance(data, list) else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image_path = self.image_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        caption = sample["caption"]
        full_text = f"{self.prompt} {caption}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        prompt_encoding = self.tokenizer(
            self.prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@DATASETS.register("grounding")
class GroundingDataset(Dataset):
    """Visual Grounding / Referring Expression dataset.

    Expected JSON format: [{"image": "path", "expression": "text", "bbox": [x1,y1,x2,y2]}, ...]
    Bounding boxes are normalized to [0, 1].
    """

    def __init__(
        self,
        data_path: str | Path,
        image_dir: str | Path,
        tokenizer: Any,
        transform: Callable | None = None,
        max_length: int = 256,
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform or build_transform(image_size=336)
        self.max_length = max_length
        self.samples = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if not self.data_path.exists():
            return []
        with open(self.data_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image_path = self.image_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        expression = sample["expression"]
        bbox = sample["bbox"]
        bbox_str = f"[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"

        prompt = f"Locate: {expression}\nBounding box:"
        full_text = f"{prompt} {bbox_str}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        prompt_encoding = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "bbox": bbox_tensor,
        }
