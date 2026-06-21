"""Tests for FlashVLM registry system."""

import pytest

from flashvlm.registry import DATASETS, MODELS, PROJECTORS, TASKS, VISION_ENCODERS, Registry


class TestRegistry:
    def test_register_and_get(self):
        registry = Registry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        assert "my_class" in registry
        assert registry.get("my_class") is MyClass

    def test_register_without_name(self):
        registry = Registry("test")

        @registry.register()
        class AnotherClass:
            pass

        assert "AnotherClass" in registry

    def test_get_missing_raises(self):
        registry = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_duplicate_registration_raises(self):
        registry = Registry("test")

        @registry.register("dup")
        class First:
            pass

        with pytest.raises(KeyError, match="already registered"):
            @registry.register("dup")
            class Second:
                pass

    def test_build(self):
        registry = Registry("test")

        @registry.register("builder")
        class Buildable:
            def __init__(self, x, y=10):
                self.x = x
                self.y = y

        obj = registry.build("builder", 5, y=20)
        assert obj.x == 5
        assert obj.y == 20

    def test_list(self):
        registry = Registry("test")

        @registry.register("alpha")
        class A:
            pass

        @registry.register("beta")
        class B:
            pass

        items = registry.list()
        assert items == ["alpha", "beta"]

    def test_len(self):
        registry = Registry("test")
        assert len(registry) == 0

        @registry.register("item")
        class Item:
            pass

        assert len(registry) == 1

    def test_repr(self):
        registry = Registry("my_registry")
        repr_str = repr(registry)
        assert "my_registry" in repr_str


class TestGlobalRegistries:
    def test_models_registry_has_entries(self):
        assert "flashvlm" in MODELS
        assert "llava" in MODELS
        assert "qwen_vl" in MODELS
        assert "internvl" in MODELS

    def test_vision_encoders_registry(self):
        assert "clip" in VISION_ENCODERS
        assert "siglip" in VISION_ENCODERS
        assert "dinov2" in VISION_ENCODERS

    def test_projectors_registry(self):
        assert "mlp" in PROJECTORS
        assert "qformer" in PROJECTORS
        assert "cross_attention" in PROJECTORS

    def test_tasks_registry(self):
        assert "vqa" in TASKS
        assert "captioning" in TASKS
        assert "grounding" in TASKS
        assert "ocr" in TASKS
        assert "reasoning" in TASKS

    def test_datasets_registry(self):
        assert "vqa" in DATASETS
        assert "captioning" in DATASETS
        assert "grounding" in DATASETS
