"""Visual Reasoning and Chain-of-Thought task module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from flashvlm.registry import TASKS


@TASKS.register("reasoning")
class ReasoningTask:
    """Visual reasoning with chain-of-thought prompting.

    Supports step-by-step reasoning, spatial reasoning, counting,
    comparison, and logical inference about images.
    """

    PROMPT_TEMPLATES = {
        "chain_of_thought": (
            "Look at the image carefully and answer the following question. "
            "Think step by step before giving your final answer.\n\n"
            "Question: {question}\n\n"
            "Let's think step by step:\n"
        ),
        "spatial": (
            "Analyze the spatial relationships between objects in this image.\n\n"
            "Question: {question}\n\n"
            "Reasoning:"
        ),
        "counting": (
            "Count the objects carefully in this image.\n\n"
            "Question: {question}\n\n"
            "Let me count step by step:"
        ),
        "comparison": (
            "Compare the elements in this image.\n\n"
            "Question: {question}\n\n"
            "Analysis:"
        ),
        "causal": (
            "Analyze the cause and effect relationships visible in this image.\n\n"
            "Question: {question}\n\n"
            "Reasoning:"
        ),
    }

    def __init__(
        self,
        model: Any,
        reasoning_type: str = "chain_of_thought",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ):
        self.model = model
        self.reasoning_type = reasoning_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def reason(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        reasoning_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Perform visual reasoning with step-by-step explanation.

        Args:
            image: Input image.
            question: Reasoning question about the image.
            reasoning_type: Type of reasoning to apply.

        Returns:
            Dictionary with 'reasoning' (steps) and 'answer' (final answer).
        """
        rtype = reasoning_type or self.reasoning_type
        prompt = self._format_prompt(question, rtype)

        response = self.model.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        return self._parse_response(response)

    def multi_step_reason(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        num_steps: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform multi-step reasoning with iterative refinement.

        Args:
            image: Input image.
            question: Question to reason about.
            num_steps: Number of reasoning iterations.

        Returns:
            Dictionary with steps and final answer.
        """
        steps = []
        context = ""

        for step in range(num_steps):
            if step == 0:
                step_prompt = (
                    f"Question: {question}\n\n"
                    f"Step {step + 1}: What do you observe in the image that's relevant?"
                )
            elif step == num_steps - 1:
                step_prompt = (
                    f"Previous observations:\n{context}\n\n"
                    f"Step {step + 1}: Based on all observations, what is the final answer to: {question}"
                )
            else:
                step_prompt = (
                    f"Previous observations:\n{context}\n\n"
                    f"Step {step + 1}: What additional details or relationships do you notice?"
                )

            response = self.model.generate(
                prompt=step_prompt,
                image=image,
                max_new_tokens=self.max_new_tokens // num_steps,
                temperature=self.temperature,
                **kwargs,
            )
            steps.append(response.strip())
            context += f"\nStep {step + 1}: {response.strip()}"

        return {
            "steps": steps,
            "reasoning": context.strip(),
            "answer": steps[-1] if steps else "",
        }

    def _format_prompt(self, question: str, reasoning_type: str) -> str:
        template = self.PROMPT_TEMPLATES.get(
            reasoning_type, self.PROMPT_TEMPLATES["chain_of_thought"]
        )
        return template.format(question=question)

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse reasoning response into steps and final answer."""
        response = response.strip()

        answer_markers = ["therefore", "final answer", "the answer is", "in conclusion", "so,"]
        reasoning = response
        answer = ""

        for marker in answer_markers:
            lower_resp = response.lower()
            if marker in lower_resp:
                idx = lower_resp.index(marker)
                reasoning = response[:idx].strip()
                answer = response[idx:].strip()
                break

        if not answer:
            lines = response.split("\n")
            if len(lines) > 1:
                answer = lines[-1].strip()
                reasoning = "\n".join(lines[:-1]).strip()
            else:
                answer = response

        return {"reasoning": reasoning, "answer": answer}

    def evaluate(
        self,
        predictions: List[Dict[str, str]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Evaluate reasoning quality."""
        correct = 0
        has_reasoning = 0

        for pred, gt in zip(predictions, ground_truths):
            answer = pred.get("answer", "").lower().strip()
            gt_lower = gt.lower().strip()
            if gt_lower in answer or answer in gt_lower:
                correct += 1
            if pred.get("reasoning", "").strip():
                has_reasoning += 1

        total = max(len(ground_truths), 1)
        return {
            "accuracy": correct / total,
            "reasoning_rate": has_reasoning / total,
        }
