"""GUI Agent task: screenshot understanding, UI grounding, action prediction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from flashvlm.registry import TASKS


@dataclass
class UIElement:
    """Represents a detected UI element with bounding box and metadata."""

    element_type: str
    text: str = ""
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 0.0
    element_id: int = 0
    interactable: bool = True


@dataclass
class GUIAction:
    """Represents an action to take on a GUI element."""

    action_type: str
    target: UIElement | None = None
    coordinates: tuple[float, float] = (0.0, 0.0)
    text_input: str = ""
    reasoning: str = ""


@TASKS.register("gui_agent")
class GUIAgentTask:
    """GUI Agent for screenshot understanding and interaction.

    Capabilities:
    - Screenshot understanding and element detection
    - UI element grounding (locating elements by description)
    - Action prediction (click, type, scroll, etc.)
    - Multi-step task planning on GUI interfaces
    """

    ACTION_TYPES = [
        "click",
        "type",
        "scroll_up",
        "scroll_down",
        "hover",
        "drag",
        "select",
        "press_key",
    ]

    PROMPTS = {
        "describe_screen": (
            "Analyze this screenshot and describe all visible UI elements, "
            "their positions, and the current state of the application.\n\n"
            "Description:"
        ),
        "ground_element": (
            "In this screenshot, locate the UI element described below. "
            "Provide the bounding box coordinates as [x1, y1, x2, y2] "
            "normalized to [0, 1].\n\n"
            "Element: {element_description}\n\nBounding box:"
        ),
        "predict_action": (
            "Given this screenshot and the following task, determine the next "
            "GUI action to take. Specify the action type ({action_types}) and "
            "the target coordinates [x, y] normalized to [0, 1].\n\n"
            "Task: {task}\n\n"
            "Action:"
        ),
        "plan_steps": (
            "Given this screenshot, create a step-by-step plan to accomplish "
            "the following task. For each step, describe the action and target.\n\n"
            "Task: {task}\n\nPlan:"
        ),
        "extract_text": (
            "Extract all visible text from this screenshot, organized by "
            "UI sections (header, body, sidebar, footer, etc.).\n\nText:"
        ),
        "identify_elements": (
            "List all interactive UI elements in this screenshot. "
            "For each element, provide: type, label/text, and approximate "
            "position as [x, y] normalized to [0, 1].\n\nElements:"
        ),
    }

    def __init__(
        self,
        model: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        screen_size: tuple[int, int] = (1920, 1080),
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.screen_size = screen_size

    def describe_screen(
        self,
        screenshot: str | Path | Image.Image,
        **kwargs: Any,
    ) -> str:
        """Describe the contents and layout of a screenshot."""
        result = self.model.generate(
            prompt=self.PROMPTS["describe_screen"],
            image=screenshot,
            max_new_tokens=self.max_new_tokens,
            temperature=0.3,
            **kwargs,
        )
        return result.strip()

    def ground_element(
        self,
        screenshot: str | Path | Image.Image,
        element_description: str,
        **kwargs: Any,
    ) -> UIElement:
        """Locate a UI element by its description.

        Args:
            screenshot: Screenshot image.
            element_description: Natural language description of the element.

        Returns:
            UIElement with bounding box coordinates.
        """
        prompt = self.PROMPTS["ground_element"].format(
            element_description=element_description,
        )
        response = self.model.generate(
            prompt=prompt,
            image=screenshot,
            max_new_tokens=128,
            temperature=0.05,
            **kwargs,
        )
        bbox = self._parse_bbox(response)
        return UIElement(
            element_type="detected",
            text=element_description,
            bbox=bbox,
            confidence=0.8 if bbox != (0.0, 0.0, 0.0, 0.0) else 0.0,
        )

    def predict_action(
        self,
        screenshot: str | Path | Image.Image,
        task: str,
        history: list[GUIAction] | None = None,
        **kwargs: Any,
    ) -> GUIAction:
        """Predict the next GUI action to accomplish a task.

        Args:
            screenshot: Current screenshot.
            task: Task description.
            history: Previous actions taken (for multi-step reasoning).

        Returns:
            Predicted GUIAction.
        """
        action_types_str = ", ".join(self.ACTION_TYPES)
        prompt = self.PROMPTS["predict_action"].format(
            task=task,
            action_types=action_types_str,
        )

        if history:
            history_str = "\n".join(
                f"  Step {i + 1}: {a.action_type} at "
                f"({a.coordinates[0]:.2f}, {a.coordinates[1]:.2f})"
                + (f' "{a.text_input}"' if a.text_input else "")
                for i, a in enumerate(history)
            )
            prompt = f"Previous actions:\n{history_str}\n\n{prompt}"

        response = self.model.generate(
            prompt=prompt,
            image=screenshot,
            max_new_tokens=256,
            temperature=0.1,
            **kwargs,
        )
        return self._parse_action(response)

    def plan_task(
        self,
        screenshot: str | Path | Image.Image,
        task: str,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Plan multi-step actions to accomplish a task.

        Args:
            screenshot: Current screenshot.
            task: Task to accomplish.

        Returns:
            List of planned steps with action and target descriptions.
        """
        prompt = self.PROMPTS["plan_steps"].format(task=task)
        response = self.model.generate(
            prompt=prompt,
            image=screenshot,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            **kwargs,
        )
        return self._parse_plan(response)

    def identify_elements(
        self,
        screenshot: str | Path | Image.Image,
        **kwargs: Any,
    ) -> list[UIElement]:
        """Identify interactive elements in a screenshot.

        Returns:
            List of detected UIElements.
        """
        response = self.model.generate(
            prompt=self.PROMPTS["identify_elements"],
            image=screenshot,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            **kwargs,
        )
        return self._parse_elements(response)

    def extract_text(
        self,
        screenshot: str | Path | Image.Image,
        **kwargs: Any,
    ) -> str:
        """Extract all visible text from a screenshot."""
        result = self.model.generate(
            prompt=self.PROMPTS["extract_text"],
            image=screenshot,
            max_new_tokens=self.max_new_tokens,
            temperature=0.05,
            **kwargs,
        )
        return result.strip()

    def execute_task(
        self,
        screenshots: list[str | Path | Image.Image],
        task: str,
        max_steps: int = 10,
        **kwargs: Any,
    ) -> list[GUIAction]:
        """Execute a multi-step GUI task, predicting actions iteratively.

        Uses the first screenshot to start, then processes subsequent
        screenshots (simulating environment feedback) step by step.
        """
        actions: list[GUIAction] = []
        for step in range(min(max_steps, len(screenshots))):
            action = self.predict_action(
                screenshots[step],
                task,
                history=actions,
                **kwargs,
            )
            actions.append(action)
            if action.action_type == "done":
                break
        return actions

    def _parse_bbox(self, response: str) -> tuple[float, float, float, float]:
        numbers = re.findall(r"[\d.]+", response)
        if len(numbers) >= 4:
            try:
                coords = [float(n) for n in numbers[:4]]
                coords = [max(0.0, min(1.0, c)) for c in coords]
                return (coords[0], coords[1], coords[2], coords[3])
            except ValueError:
                pass
        return (0.0, 0.0, 0.0, 0.0)

    def _parse_action(self, response: str) -> GUIAction:
        response_lower = response.lower()
        action_type = "click"
        for at in self.ACTION_TYPES:
            if at in response_lower:
                action_type = at
                break

        coords = re.findall(r"[\d.]+", response)
        x, y = 0.5, 0.5
        if len(coords) >= 2:
            try:
                x = max(0.0, min(1.0, float(coords[0])))
                y = max(0.0, min(1.0, float(coords[1])))
            except ValueError:
                pass

        text_input = ""
        text_match = re.search(r'"([^"]*)"', response)
        if text_match:
            text_input = text_match.group(1)

        return GUIAction(
            action_type=action_type,
            coordinates=(x, y),
            text_input=text_input,
            reasoning=response.strip(),
        )

    def _parse_plan(self, response: str) -> list[dict[str, str]]:
        steps = []
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            step_match = re.match(r"(?:\d+[\.\)]\s*)?(.+)", line)
            if step_match and line:
                steps.append(
                    {
                        "step": str(len(steps) + 1),
                        "description": step_match.group(1).strip(),
                    }
                )
        return steps

    def _parse_elements(self, response: str) -> list[UIElement]:
        elements = []
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            coords = re.findall(r"[\d.]+", line)
            x, y = 0.5, 0.5
            if len(coords) >= 2:
                try:
                    x, y = float(coords[0]), float(coords[1])
                except ValueError:
                    pass
            elem_type = "button"
            for t in ["button", "input", "link", "text", "checkbox", "dropdown", "icon", "menu"]:
                if t in line.lower():
                    elem_type = t
                    break
            elements.append(
                UIElement(
                    element_type=elem_type,
                    text=line,
                    bbox=(x, y, x + 0.05, y + 0.03),
                    element_id=i,
                )
            )
        return elements
