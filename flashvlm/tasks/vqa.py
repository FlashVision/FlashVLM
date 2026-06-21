"""Visual Question Answering task module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from flashvlm.registry import TASKS


@TASKS.register("vqa")
class VQATask:
    """Visual Question Answering task.

    Supports open-ended VQA, multiple-choice VQA, and yes/no questions.
    """

    PROMPT_TEMPLATES = {
        "default": "Question: {question}\nAnswer:",
        "llava": "USER: <image>\n{question}\nASSISTANT:",
        "short": "{question}\nShort answer:",
        "detailed": "Look at the image carefully and answer the following question in detail.\n\nQuestion: {question}\n\nAnswer:",
        "multiple_choice": "Question: {question}\nOptions:\n{options}\nAnswer:",
    }

    def __init__(
        self,
        model: Any,
        prompt_style: str = "default",
        max_new_tokens: int = 128,
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
        options: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Answer a visual question.

        Args:
            image: Input image.
            question: Question about the image.
            options: Optional multiple-choice options.

        Returns:
            The model's answer.
        """
        prompt = self._format_prompt(question, options)
        answer = self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        return self._postprocess_answer(answer)

    def batch_answer(
        self,
        images: List[Union[str, Path, Image.Image]],
        questions: List[str],
        **kwargs: Any,
    ) -> List[str]:
        """Answer multiple VQA questions."""
        answers = []
        for image, question in zip(images, questions):
            answers.append(self.answer(image, question, **kwargs))
        return answers

    def _format_prompt(self, question: str, options: Optional[List[str]] = None) -> str:
        if options and self.prompt_style == "multiple_choice":
            options_str = "\n".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options))
            return self.PROMPT_TEMPLATES["multiple_choice"].format(
                question=question, options=options_str
            )
        template = self.PROMPT_TEMPLATES.get(self.prompt_style, self.PROMPT_TEMPLATES["default"])
        return template.format(question=question)

    def _postprocess_answer(self, answer: str) -> str:
        """Clean up model output."""
        answer = answer.strip()
        for stop in ["\n", "Question:", "USER:", "ASSISTANT:"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]
        return answer.strip()

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[Union[str, List[str]]],
    ) -> Dict[str, float]:
        """Compute VQA accuracy metrics.

        Uses soft accuracy: a prediction is correct if it matches
        any of the ground truth answers (case-insensitive).
        """
        correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            if isinstance(gt, str):
                gt = [gt]
            pred_normalized = pred.lower().strip().rstrip(".")
            gt_normalized = [g.lower().strip().rstrip(".") for g in gt]
            if pred_normalized in gt_normalized:
                correct += 1

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "correct": correct, "total": total}
