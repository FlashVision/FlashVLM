"""Task-specific modules for FlashVLM."""

from flashvlm.tasks.vqa import VQATask
from flashvlm.tasks.captioning import CaptioningTask
from flashvlm.tasks.grounding import GroundingTask
from flashvlm.tasks.ocr import OCRTask
from flashvlm.tasks.reasoning import ReasoningTask

__all__ = ["VQATask", "CaptioningTask", "GroundingTask", "OCRTask", "ReasoningTask"]
