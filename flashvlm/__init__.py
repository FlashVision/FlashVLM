"""
FlashVLM - High-Performance Vision-Language Models for Multimodal AI.

A unified framework for training, fine-tuning, and deploying Vision-Language Models
supporting LLaVA, Qwen-VL, InternVL, and Phi-Vision architectures.
"""

__version__ = "1.0.0"
__author__ = "FlashVision Team"

from flashvlm.models.vlm import FlashVLM
from flashvlm.engine.trainer import Trainer
from flashvlm.engine.predictor import Predictor
from flashvlm.engine.exporter import Exporter
from flashvlm.models.lora import apply_lora
from flashvlm.cfg.config import get_config
from flashvlm.solutions.multimodal_chat import MultimodalChat
from flashvlm.solutions.image_analyzer import ImageAnalyzer
from flashvlm.analytics.benchmark import Benchmark

import flashvlm.models.architectures  # noqa: F401 — register architectures
import flashvlm.data.datasets  # noqa: F401 — register datasets

__all__ = [
    "FlashVLM",
    "Trainer",
    "Predictor",
    "Exporter",
    "apply_lora",
    "get_config",
    "MultimodalChat",
    "ImageAnalyzer",
    "Benchmark",
    "__version__",
]
