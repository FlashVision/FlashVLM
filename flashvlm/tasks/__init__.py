"""Task-specific modules for FlashVLM."""

from flashvlm.tasks.vqa import VQATask
from flashvlm.tasks.captioning import CaptioningTask
from flashvlm.tasks.grounding import GroundingTask
from flashvlm.tasks.ocr import OCRTask
from flashvlm.tasks.reasoning import ReasoningTask
from flashvlm.tasks.chart_qa import ChartQATask
from flashvlm.tasks.gui_agent import GUIAgentTask

__all__ = [
    "VQATask",
    "CaptioningTask",
    "GroundingTask",
    "OCRTask",
    "ReasoningTask",
    "ChartQATask",
    "GUIAgentTask",
]
