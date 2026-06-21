"""Interactive multimodal chatbot solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class MultimodalChat:
    """Interactive multimodal chatbot with conversation history.

    Supports text-only and image+text conversations with context tracking.
    """

    def __init__(
        self,
        model_name: str = "llava-v1.5-7b",
        device: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a helpful multimodal AI assistant. You can see and understand images, "
            "answer questions about them, and engage in natural conversation."
        )

        self.conversation_history: List[Dict[str, Any]] = []
        self.images: List[Any] = []
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the VLM model."""
        try:
            from flashvlm.models.vlm import FlashVLM
            self.model = FlashVLM.from_pretrained(self.model_name, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load model '{self.model_name}': {e}")
            self.model = None

    def add_image(self, image: Union[str, Path, Image.Image]) -> None:
        """Add an image to the current conversation context.

        Args:
            image: Image path, URL, or PIL Image object.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                image = Image.open(path).convert("RGB")

        self.images.append(image)
        self.conversation_history.append({
            "role": "user",
            "type": "image",
            "content": f"[Image added: {len(self.images)}]",
        })

    def send(self, message: str, **kwargs: Any) -> str:
        """Send a message and get a response.

        Args:
            message: User text message.

        Returns:
            Assistant's response text.
        """
        self.conversation_history.append({
            "role": "user",
            "type": "text",
            "content": message,
        })

        prompt = self._build_prompt()
        current_image = self.images[-1] if self.images else None

        if self.model is not None:
            response = self.model.generate(
                prompt=prompt,
                image=current_image,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs,
            )
        else:
            response = f"[Model not loaded. Would respond to: {message}]"

        self.conversation_history.append({
            "role": "assistant",
            "type": "text",
            "content": response,
        })

        return response

    def _build_prompt(self) -> str:
        """Build a conversational prompt from history."""
        parts = [f"System: {self.system_prompt}\n"]

        for turn in self.conversation_history[-10:]:
            role = turn["role"].capitalize()
            content = turn["content"]
            if turn["type"] == "image":
                parts.append(f"{role}: <image>")
            else:
                parts.append(f"{role}: {content}")

        parts.append("Assistant:")
        return "\n".join(parts)

    def reset(self) -> None:
        """Clear conversation history and images."""
        self.conversation_history = []
        self.images = []

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = prompt

    @property
    def num_turns(self) -> int:
        """Number of conversation turns."""
        return sum(1 for t in self.conversation_history if t["role"] == "user" and t["type"] == "text")
