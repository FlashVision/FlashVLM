"""Benchmarking suite for VLM evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


class Benchmark:
    """Comprehensive VLM benchmarking suite.

    Supports standard benchmarks: VQAv2, GQA, TextVQA, MMBench, POPE, etc.
    """

    SUPPORTED_DATASETS = {
        "vqav2": {"metrics": ["accuracy"], "split": "val"},
        "gqa": {"metrics": ["accuracy"], "split": "testdev"},
        "textvqa": {"metrics": ["accuracy"], "split": "val"},
        "mmbench": {"metrics": ["accuracy"], "split": "test"},
        "pope": {"metrics": ["accuracy", "f1"], "split": "test"},
        "seedbench": {"metrics": ["accuracy"], "split": "test"},
    }

    def __init__(
        self,
        model: nn.Module,
        dataset: str = "vqav2",
        split: str = "val",
        data_path: str | None = None,
    ):
        self.model = model
        self.dataset = dataset
        self.split = split
        self.data_path = data_path
        self.results: dict[str, Any] = {}

    def run(
        self,
        batch_size: int = 8,
        max_samples: int | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Run the benchmark evaluation.

        Args:
            batch_size: Inference batch size.
            max_samples: Limit number of samples (for quick evaluation).
            verbose: Print progress.

        Returns:
            Dictionary of metric names to scores.
        """
        if verbose:
            print(f"Running benchmark: {self.dataset} ({self.split})")
            print(f"  Batch size: {batch_size}")

        start_time = time.time()

        samples = self._load_benchmark_data(max_samples)
        if not samples:
            print(f"No data found for {self.dataset}. Running synthetic benchmark.")
            return self._run_synthetic_benchmark()

        predictions = self._generate_predictions(samples, batch_size, verbose)
        metrics = self._compute_metrics(predictions, samples)

        elapsed = time.time() - start_time
        metrics["inference_time_s"] = round(elapsed, 2)
        metrics["samples_per_second"] = round(len(samples) / elapsed, 2)

        self.results = metrics
        return metrics

    def _load_benchmark_data(self, max_samples: int | None) -> list[dict[str, Any]]:
        """Load benchmark dataset."""
        if self.data_path:
            path = Path(self.data_path)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = data.get("data", data.get("questions", []))
                if max_samples:
                    samples = samples[:max_samples]
                return samples
        return []

    def _generate_predictions(
        self,
        samples: list[dict[str, Any]],
        batch_size: int,
        verbose: bool,
    ) -> list[str]:
        """Generate model predictions for benchmark samples."""
        predictions = []
        iterator = range(0, len(samples), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Evaluating")

        self.model.eval()
        with torch.no_grad():
            for i in iterator:
                batch = samples[i : i + batch_size]
                for sample in batch:
                    question = sample.get("question", "")
                    image = sample.get("image")

                    if hasattr(self.model, "ask"):
                        pred = self.model.ask(question, image=image, max_new_tokens=32)
                    elif hasattr(self.model, "generate"):
                        pred = self.model.generate(prompt=question, image=image, max_new_tokens=32)
                    else:
                        pred = ""
                    predictions.append(pred)

        return predictions

    def _compute_metrics(
        self,
        predictions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute evaluation metrics based on dataset type."""
        from flashvlm.analytics.metrics import vqa_accuracy

        ground_truths = []
        for sample in samples:
            answer = sample.get("answer", sample.get("answers", ""))
            if isinstance(answer, str):
                ground_truths.append([answer])
            elif isinstance(answer, list):
                ground_truths.append(answer)
            else:
                ground_truths.append([""])

        accuracy = vqa_accuracy(predictions, ground_truths)
        return {"accuracy": round(accuracy, 4)}

    def _run_synthetic_benchmark(self) -> dict[str, float]:
        """Run a synthetic performance benchmark when no data is available."""
        print("Running performance benchmark (latency, throughput)...")

        device = next(self.model.parameters()).device
        img_size = 336
        num_samples = 50
        latencies = []

        for _ in tqdm(range(num_samples), desc="Benchmarking"):
            dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                if hasattr(self.model, "encode_image"):
                    self.model.encode_image(dummy_input)
                elif hasattr(self.model, "vision_encoder"):
                    self.model.vision_encoder(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        throughput = 1.0 / avg_latency

        return {
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "throughput_fps": round(throughput, 2),
            "num_samples": num_samples,
        }

    def summary(self) -> str:
        """Generate a summary of benchmark results."""
        lines = [f"Benchmark: {self.dataset} ({self.split})", "-" * 40]
        for metric, value in self.results.items():
            lines.append(f"  {metric}: {value}")
        return "\n".join(lines)
