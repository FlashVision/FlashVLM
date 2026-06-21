"""Benchmark evaluation suite: VQAv2, TextVQA, MMBench, POPE loaders and scoring."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image


class BaseEvaluator(ABC):
    """Base evaluator for VLM benchmarks."""

    name: str = "base"

    def __init__(
        self,
        model: nn.Module,
        data_path: str | None = None,
        image_dir: str | None = None,
        split: str = "val",
        max_new_tokens: int = 64,
    ):
        self.model = model
        self.data_path = data_path
        self.image_dir = image_dir
        self.split = split
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        max_samples: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run full evaluation pipeline."""
        start = time.time()
        samples = self.load_data(max_samples)
        if not samples:
            if verbose:
                print(
                    f"[{self.name}] No data found at {self.data_path}. "
                    "Running with synthetic samples."
                )
            samples = self._generate_synthetic_samples()

        predictions = self._run_inference(samples, verbose)
        metrics = self.compute_metrics(predictions, samples)
        elapsed = time.time() - start
        metrics["elapsed_s"] = round(elapsed, 2)
        metrics["num_samples"] = len(samples)
        metrics["benchmark"] = self.name
        if verbose:
            self._print_results(metrics)
        return metrics

    @abstractmethod
    def load_data(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        """Load benchmark samples."""
        ...

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any]) -> str:
        """Format the prompt for this benchmark."""
        ...

    @abstractmethod
    def compute_metrics(
        self,
        predictions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute benchmark-specific metrics."""
        ...

    def _run_inference(
        self,
        samples: list[dict[str, Any]],
        verbose: bool,
    ) -> list[str]:
        predictions = []
        self.model.eval()
        total = len(samples)
        with torch.no_grad():
            for i, sample in enumerate(samples):
                if verbose and (i + 1) % max(1, total // 10) == 0:
                    print(f"  [{self.name}] {i + 1}/{total}")
                prompt = self.format_prompt(sample)
                image = self._load_image(sample)
                if hasattr(self.model, "generate"):
                    pred = self.model.generate(
                        prompt=prompt,
                        image=image,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.0,
                    )
                elif hasattr(self.model, "ask"):
                    pred = self.model.ask(
                        sample.get("question", ""),
                        image=image,
                        max_new_tokens=self.max_new_tokens,
                    )
                else:
                    pred = ""
                predictions.append(pred.strip())
        return predictions

    def _load_image(self, sample: dict[str, Any]) -> Image.Image | None:
        img_path = sample.get("image") or sample.get("image_path")
        if img_path is None:
            return None
        if isinstance(img_path, Image.Image):
            return img_path
        if self.image_dir:
            full_path = Path(self.image_dir) / img_path
        else:
            full_path = Path(img_path)
        if full_path.exists():
            return Image.open(full_path).convert("RGB")
        return None

    def _load_json(self, path: str | None, max_samples: int | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        p = Path(path)
        if not p.exists():
            return []
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("questions", data.get("annotations", [])))
        if max_samples:
            data = data[:max_samples]
        return data

    def _generate_synthetic_samples(self) -> list[dict[str, Any]]:
        return [
            {"question": "What color is the sky?", "answer": "blue", "answers": ["blue"]},
            {"question": "How many objects?", "answer": "3", "answers": ["3", "three"]},
        ]

    def _print_results(self, metrics: dict[str, Any]) -> None:
        print(f"\n{'=' * 50}")
        print(f"Benchmark: {metrics.get('benchmark', self.name)}")
        print(f"{'=' * 50}")
        for k, v in metrics.items():
            if k not in ("benchmark",):
                print(f"  {k}: {v}")
        print()


class VQAv2Evaluator(BaseEvaluator):
    """VQAv2 benchmark evaluator with soft accuracy scoring."""

    name = "vqav2"

    def load_data(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        samples = self._load_json(self.data_path, max_samples)
        parsed = []
        for s in samples:
            question = s.get("question", "")
            answers = s.get("answers", [])
            if isinstance(answers, list) and answers and isinstance(answers[0], dict):
                answers = [a.get("answer", "") for a in answers]
            elif isinstance(answers, str):
                answers = [answers]
            image = s.get("image", s.get("image_path", s.get("image_id", "")))
            if isinstance(image, int):
                image = f"COCO_val2014_{image:012d}.jpg"
            parsed.append({"question": question, "answers": answers, "image": image})
        return parsed

    def format_prompt(self, sample: dict[str, Any]) -> str:
        q = sample["question"]
        return f"Question: {q}\nShort answer:"

    def compute_metrics(
        self,
        predictions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        correct = 0.0
        for pred, sample in zip(predictions, samples):
            answers = sample.get("answers", [])
            if not answers:
                continue
            pred_norm = _normalize_vqa(pred)
            answer_counts: dict[str, int] = {}
            for a in answers:
                a_norm = _normalize_vqa(a)
                answer_counts[a_norm] = answer_counts.get(a_norm, 0) + 1
            if pred_norm in answer_counts:
                correct += min(answer_counts[pred_norm] / 3.0, 1.0)
        accuracy = correct / max(len(predictions), 1)
        return {"accuracy": round(accuracy, 4)}


class TextVQAEvaluator(BaseEvaluator):
    """TextVQA benchmark evaluator for OCR-dependent visual QA."""

    name = "textvqa"

    def load_data(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        samples = self._load_json(self.data_path, max_samples)
        parsed = []
        for s in samples:
            q = s.get("question", "")
            answers = s.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            image = s.get("image", s.get("image_path", s.get("image_id", "")))
            ocr_tokens = s.get("ocr_tokens", [])
            parsed.append(
                {
                    "question": q,
                    "answers": answers,
                    "image": image,
                    "ocr_tokens": ocr_tokens,
                }
            )
        return parsed

    def format_prompt(self, sample: dict[str, Any]) -> str:
        q = sample["question"]
        ocr = sample.get("ocr_tokens", [])
        if ocr:
            ocr_str = ", ".join(ocr[:20])
            return f"OCR tokens in the image: {ocr_str}\nQuestion: {q}\nShort answer:"
        return f"Read the text in the image and answer: {q}\nAnswer:"

    def compute_metrics(
        self,
        predictions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        correct = 0.0
        for pred, sample in zip(predictions, samples):
            answers = sample.get("answers", [])
            pred_norm = _normalize_vqa(pred)
            answer_counts: dict[str, int] = {}
            for a in answers:
                a_norm = _normalize_vqa(a)
                answer_counts[a_norm] = answer_counts.get(a_norm, 0) + 1
            if pred_norm in answer_counts:
                correct += min(answer_counts[pred_norm] / 3.0, 1.0)
        return {"accuracy": round(correct / max(len(predictions), 1), 4)}


class MMBenchEvaluator(BaseEvaluator):
    """MMBench benchmark evaluator for multi-choice multimodal evaluation."""

    name = "mmbench"

    def load_data(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        samples = self._load_json(self.data_path, max_samples)
        parsed = []
        for s in samples:
            q = s.get("question", "")
            options = {}
            for key in ["A", "B", "C", "D"]:
                if key in s:
                    options[key] = s[key]
            answer = s.get("answer", "")
            image = s.get("image", s.get("image_path", ""))
            parsed.append(
                {
                    "question": q,
                    "options": options,
                    "answer": answer,
                    "image": image,
                }
            )
        return parsed

    def format_prompt(self, sample: dict[str, Any]) -> str:
        q = sample["question"]
        options = sample.get("options", {})
        opts_str = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
        return f"Question: {q}\n{opts_str}\nAnswer with the option letter (A, B, C, or D):"

    def compute_metrics(
        self,
        predictions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        correct = 0
        for pred, sample in zip(predictions, samples):
            gt = sample.get("answer", "").strip().upper()
            pred_letter = self._extract_choice(pred)
            if pred_letter == gt:
                correct += 1
        return {"accuracy": round(correct / max(len(predictions), 1), 4)}

    @staticmethod
    def _extract_choice(text: str) -> str:
        text = text.strip()
        if text and text[0].upper() in "ABCD":
            return text[0].upper()
        match = re.search(r"\b([ABCD])\b", text.upper())
        if match:
            return match.group(1)
        return text[:1].upper() if text else ""


class POPEEvaluator(BaseEvaluator):
    """POPE benchmark evaluator for object hallucination detection."""

    name = "pope"

    def load_data(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        if self.data_path is None:
            return []
        p = Path(self.data_path)
        if not p.exists():
            return []

        samples = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                samples.append(
                    {
                        "question": item.get("text", item.get("question", "")),
                        "answer": item.get("label", item.get("answer", "")),
                        "image": item.get("image", ""),
                    }
                )
        if max_samples:
            samples = samples[:max_samples]
        return samples

    def format_prompt(self, sample: dict[str, Any]) -> str:
        q = sample["question"]
        return f"{q}\nAnswer with yes or no:"

    def compute_metrics(
        self,
        predictions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        tp = fp = tn = fn = 0
        for pred, sample in zip(predictions, samples):
            gt = sample.get("answer", "").lower().strip()
            pred_yn = "yes" if "yes" in pred.lower() else "no"
            gt_yn = "yes" if "yes" in gt else "no"

            if pred_yn == "yes" and gt_yn == "yes":
                tp += 1
            elif pred_yn == "yes" and gt_yn == "no":
                fp += 1
            elif pred_yn == "no" and gt_yn == "no":
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "yes_ratio": round((tp + fp) / max(total, 1), 4),
        }


def _normalize_vqa(text: str) -> str:
    """Normalize VQA answer text for soft accuracy matching."""
    text = text.lower().strip()
    text = text.rstrip(".")
    for punct in ["?", "!", ",", ";", ":"]:
        text = text.replace(punct, "")
    articles = ["a ", "an ", "the "]
    for art in articles:
        if text.startswith(art):
            text = text[len(art) :]
    return text.strip()


def run_evaluation(
    model: nn.Module,
    benchmarks: list[str],
    data_dir: str | None = None,
    image_dir: str | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run multiple benchmark evaluations.

    Args:
        model: VLM model with generate() or ask() method.
        benchmarks: List of benchmark names ('vqav2', 'textvqa', 'mmbench', 'pope').
        data_dir: Base directory for benchmark data files.
        image_dir: Base directory for images.
        max_samples: Limit samples per benchmark.
        verbose: Print progress.

    Returns:
        Dict mapping benchmark name to its metrics.
    """
    evaluator_map = {
        "vqav2": VQAv2Evaluator,
        "textvqa": TextVQAEvaluator,
        "mmbench": MMBenchEvaluator,
        "pope": POPEEvaluator,
    }

    results = {}
    for bench_name in benchmarks:
        cls = evaluator_map.get(bench_name)
        if cls is None:
            print(f"Unknown benchmark: {bench_name}. Skipping.")
            continue

        data_path = None
        img_dir = image_dir
        if data_dir:
            candidates = [
                Path(data_dir) / f"{bench_name}.json",
                Path(data_dir) / bench_name / "val.json",
                Path(data_dir) / bench_name / "test.jsonl",
            ]
            for c in candidates:
                if c.exists():
                    data_path = str(c)
                    break

        evaluator = cls(
            model=model,
            data_path=data_path,
            image_dir=img_dir,
            max_new_tokens=64,
        )
        results[bench_name] = evaluator.run(max_samples=max_samples, verbose=verbose)

    if verbose and len(results) > 1:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for name, metrics in results.items():
            acc = metrics.get("accuracy", "N/A")
            print(f"  {name:12s}: accuracy={acc}")
        print()

    return results
