"""Chart and Table Question Answering task module."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from flashvlm.registry import TASKS


@TASKS.register("chart_qa")
class ChartQATask:
    """Chart and Table Question Answering.

    Supports structured understanding of charts (bar, line, pie, scatter)
    and tables with value extraction, comparison, and trend analysis.
    """

    CHART_PROMPTS = {
        "value_extraction": (
            "Look at the chart in the image carefully. "
            "Answer the following question by extracting the exact value.\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        "comparison": (
            "Analyze the chart in the image. Compare the data points and "
            "answer the question.\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        "trend": (
            "Study the trend shown in the chart. "
            "Answer the following question about the trend.\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        "table_parse": (
            "Read the table in the image carefully. "
            "Extract the requested information.\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        "summarize": (
            "Provide a concise summary of the data presented in "
            "this chart/table.\n\nSummary:"
        ),
        "default": "Question: {question}\nAnswer:",
    }

    def __init__(
        self,
        model: Any,
        prompt_style: str = "value_extraction",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        self.model = model
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def answer(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        chart_type: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Answer a question about a chart or table.

        Args:
            image: Input image of a chart or table.
            question: Question about the chart/table data.
            chart_type: Optional hint for chart type (bar, line, pie, table).

        Returns:
            Extracted answer string.
        """
        style = self._infer_prompt_style(question, chart_type)
        prompt = self._format_prompt(question, style)
        answer = self.model.generate(
            prompt=prompt, image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, **kwargs,
        )
        return self._postprocess(answer)

    def parse_table(
        self,
        image: Union[str, Path, Image.Image],
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> str:
        """Extract table structure from an image.

        Args:
            image: Image containing a table.
            output_format: Output format ('markdown', 'csv', 'json').

        Returns:
            Table in requested format.
        """
        prompt_map = {
            "markdown": (
                "Extract the table from this image and format it as a "
                "markdown table with proper headers and alignment.\n\nTable:"
            ),
            "csv": (
                "Extract the table from this image and format it as CSV "
                "with comma-separated values. First row should be headers.\n\nCSV:"
            ),
            "json": (
                "Extract the table from this image and format it as a "
                "JSON array of objects where keys are column headers.\n\nJSON:"
            ),
        }
        prompt = prompt_map.get(output_format, prompt_map["markdown"])
        result = self.model.generate(
            prompt=prompt, image=image,
            max_new_tokens=512, temperature=0.05, **kwargs,
        )
        return result.strip()

    def describe_chart(
        self,
        image: Union[str, Path, Image.Image],
        detail_level: str = "detailed",
        **kwargs: Any,
    ) -> str:
        """Generate a natural language description of a chart.

        Args:
            image: Chart image.
            detail_level: 'brief' or 'detailed'.

        Returns:
            Chart description.
        """
        if detail_level == "brief":
            prompt = (
                "Briefly describe what this chart shows, including the type "
                "of chart and the main data trend.\n\nDescription:"
            )
        else:
            prompt = (
                "Provide a detailed description of this chart, including:\n"
                "1. Chart type\n2. Axis labels and units\n"
                "3. Data values or ranges\n4. Key trends or patterns\n"
                "5. Any notable observations\n\nDescription:"
            )
        result = self.model.generate(
            prompt=prompt, image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=0.3, **kwargs,
        )
        return result.strip()

    def batch_answer(
        self,
        images: List[Union[str, Path, Image.Image]],
        questions: List[str],
        **kwargs: Any,
    ) -> List[str]:
        """Answer multiple chart/table questions."""
        return [self.answer(img, q, **kwargs) for img, q in zip(images, questions)]

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[Union[str, List[str]]],
        relaxed_accuracy: bool = True,
    ) -> Dict[str, float]:
        """Compute ChartQA evaluation metrics.

        Uses relaxed accuracy: prediction within 5% of numeric ground truth
        counts as correct.
        """
        correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            if isinstance(gt, str):
                gt = [gt]
            pred_clean = self._normalize_answer(pred)
            gt_clean = [self._normalize_answer(g) for g in gt]

            if pred_clean in gt_clean:
                correct += 1
            elif relaxed_accuracy:
                pred_num = self._extract_number(pred_clean)
                if pred_num is not None:
                    for g in gt_clean:
                        gt_num = self._extract_number(g)
                        if gt_num is not None and gt_num != 0:
                            if abs(pred_num - gt_num) / abs(gt_num) <= 0.05:
                                correct += 1
                                break

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "correct": correct, "total": total}

    def _infer_prompt_style(
        self, question: str, chart_type: Optional[str]
    ) -> str:
        q_lower = question.lower()
        if chart_type == "table":
            return "table_parse"
        if any(w in q_lower for w in ["trend", "increasing", "decreasing", "over time"]):
            return "trend"
        if any(w in q_lower for w in ["compare", "difference", "more than", "less than"]):
            return "comparison"
        if any(w in q_lower for w in ["value", "how much", "how many", "what is the"]):
            return "value_extraction"
        return self.prompt_style

    def _format_prompt(self, question: str, style: str) -> str:
        template = self.CHART_PROMPTS.get(style, self.CHART_PROMPTS["default"])
        return template.format(question=question)

    def _postprocess(self, answer: str) -> str:
        answer = answer.strip()
        for stop in ["\n\n", "Question:", "Image:"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]
        return answer.strip()

    @staticmethod
    def _normalize_answer(text: str) -> str:
        text = text.lower().strip().rstrip(".")
        text = text.replace(",", "").replace("$", "").replace("%", "")
        return text

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None
