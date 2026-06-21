"""Global registry for FlashVLM components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A registry that maps string keys to classes or factory functions.

    Supports decorator-based registration and namespaced lookups.
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        """Register a class or function with an optional custom name."""

        def decorator(obj: Any) -> Any:
            key = name if name is not None else obj.__name__
            if key in self._registry:
                raise KeyError(
                    f"'{key}' is already registered in {self._name}. "
                    f"Existing: {self._registry[key]}, New: {obj}"
                )
            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        """Retrieve a registered object by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in registry '{self._name}'. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered class with given arguments."""
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list(self) -> list[str]:
        """Return all registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={self.list()})"


MODELS = Registry("models")
VISION_ENCODERS = Registry("vision_encoders")
PROJECTORS = Registry("projectors")
TASKS = Registry("tasks")
DATASETS = Registry("datasets")
TEMPLATES = Registry("templates")


def list_available() -> Dict[str, list[str]]:
    """List all registered components across all registries."""
    return {
        "models": MODELS.list(),
        "vision_encoders": VISION_ENCODERS.list(),
        "projectors": PROJECTORS.list(),
        "tasks": TASKS.list(),
        "datasets": DATASETS.list(),
        "templates": TEMPLATES.list(),
    }
