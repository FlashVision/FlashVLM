"""Tests for new FlashVLM P0 components."""

import pytest
import torch
import torch.nn as nn

from flashvlm.cfg.config import FlashVLMConfig


class TestMultiImageInput:
    def test_encode_images_returns_list(self):
        from flashvlm.models.vlm import FlashVLM

        config = FlashVLMConfig()
        config.language.model_name = "placeholder"
        model = FlashVLM(config)

        dummy_imgs = [torch.randn(1, 3, 336, 336) for _ in range(3)]
        embeds = model.encode_images(dummy_imgs)
        assert len(embeds) == 3
        for e in embeds:
            assert e.dim() == 3

    def test_interleave_multi_image_fallback(self):
        from flashvlm.models.vlm import FlashVLM

        config = FlashVLMConfig()
        config.language.model_name = "placeholder"
        model = FlashVLM(config)

        input_ids = torch.randint(0, 100, (1, 10))
        embeds = [torch.randn(1, 5, 4096), torch.randn(1, 5, 4096)]
        merged = model._interleave_multi_image_tokens(input_ids, embeds)
        assert merged.shape[1] > 10


class TestVideoEncoder:
    def test_sample_frames_uniform(self):
        from flashvlm.models.video_encoder import sample_frames

        indices = sample_frames(100, num_sample=8, strategy="uniform")
        assert len(indices) == 8
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_sample_frames_short_video(self):
        from flashvlm.models.video_encoder import sample_frames

        indices = sample_frames(3, num_sample=8)
        assert len(indices) == 3

    def test_temporal_pooling_mean(self):
        from flashvlm.models.video_encoder import TemporalPooling

        pool = TemporalPooling(64, pool_type="mean")
        x = torch.randn(2, 8, 16, 64)
        out = pool(x)
        assert out.shape == (2, 16, 64)

    def test_temporal_pooling_attention(self):
        from flashvlm.models.video_encoder import TemporalPooling

        pool = TemporalPooling(64, num_heads=4, pool_type="attention")
        x = torch.randn(1, 4, 8, 64)
        out = pool(x)
        assert out.shape == (1, 8, 64)

    def test_video_token_generator(self):
        from flashvlm.models.video_encoder import VideoTokenGenerator

        gen = VideoTokenGenerator(64, max_frames=16)
        x = torch.randn(2, 4, 8, 64)
        out = gen(x, num_frames=4)
        assert out.shape == (2, 4, 8, 64)


class TestPhiVision:
    def test_phi_vision_registered(self):
        from flashvlm.registry import MODELS

        assert "phi_vision" in MODELS

    def test_phi_image_embedding(self):
        from flashvlm.models.architectures.phi_vision import PhiImageEmbedding

        config = FlashVLMConfig()
        config.vision.hidden_size = 64
        config.language.hidden_size = 128
        config.vision.patch_size = 14
        config.vision.image_size = 28
        embed = PhiImageEmbedding(config)
        x = torch.randn(1, 3, 28, 28)
        out = embed(x)
        assert out.dim() == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 128


class TestBenchmarkEvaluation:
    def test_vqav2_evaluator_synthetic(self):
        from flashvlm.eval.evaluator import VQAv2Evaluator

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(1))

            def generate(self, prompt, image=None, **kwargs):
                return "blue"

        model = DummyModel()
        evaluator = VQAv2Evaluator(model=model)
        metrics = evaluator.run(max_samples=2, verbose=False)
        assert "accuracy" in metrics
        assert "elapsed_s" in metrics

    def test_pope_evaluator(self):
        from flashvlm.eval.evaluator import POPEEvaluator

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(1))

            def generate(self, prompt, image=None, **kwargs):
                return "yes"

        model = DummyModel()
        evaluator = POPEEvaluator(model=model)
        metrics = evaluator.compute_metrics(
            ["yes", "no", "yes"],
            [
                {"answer": "yes"},
                {"answer": "no"},
                {"answer": "no"},
            ],
        )
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_mmbench_extract_choice(self):
        from flashvlm.eval.evaluator import MMBenchEvaluator

        assert MMBenchEvaluator._extract_choice("A") == "A"
        assert MMBenchEvaluator._extract_choice("B. cats") == "B"
        assert MMBenchEvaluator._extract_choice("The answer is C") == "C"


class TestMultiStageTraining:
    def test_multi_stage_trainer_freeze_stage1(self):
        from flashvlm.training.multi_stage import MultiStageTrainer

        class SimpleVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Linear(10, 10)
                self.projector = nn.Linear(10, 10)
                self.language_model = nn.Linear(10, 10)

        model = SimpleVLM()
        config = FlashVLMConfig()
        trainer = MultiStageTrainer(model, config, output_dir="/tmp/test_multistage")
        trainer._freeze_for_stage1()

        assert not model.vision_encoder.weight.requires_grad
        assert model.projector.weight.requires_grad
        assert not model.language_model.weight.requires_grad

    def test_multi_stage_trainer_freeze_stage2(self):
        from flashvlm.training.multi_stage import MultiStageTrainer

        class SimpleVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Linear(10, 10)
                self.projector = nn.Linear(10, 10)
                self.language_model = nn.Linear(10, 10)

        model = SimpleVLM()
        config = FlashVLMConfig()
        trainer = MultiStageTrainer(model, config, output_dir="/tmp/test_multistage")
        trainer._freeze_for_stage2()

        assert not model.vision_encoder.weight.requires_grad
        assert model.projector.weight.requires_grad
        assert model.language_model.weight.requires_grad


class TestChartQA:
    def test_chart_qa_registered(self):
        from flashvlm.registry import TASKS

        assert "chart_qa" in TASKS

    def test_normalize_answer(self):
        from flashvlm.tasks.chart_qa import ChartQATask

        assert ChartQATask._normalize_answer("$1,234.5") == "1234.5"
        assert ChartQATask._normalize_answer("50%") == "50"

    def test_extract_number(self):
        from flashvlm.tasks.chart_qa import ChartQATask

        assert ChartQATask._extract_number("about 42.5 items") == 42.5
        assert ChartQATask._extract_number("no number here") is None

    def test_evaluate_relaxed(self):
        from flashvlm.tasks.chart_qa import ChartQATask

        task = ChartQATask(model=None)
        metrics = task.evaluate(["42", "100"], ["42", "105"], relaxed_accuracy=True)
        assert metrics["correct"] == 2
        assert metrics["accuracy"] == 1.0


class TestGUIAgent:
    def test_gui_agent_registered(self):
        from flashvlm.registry import TASKS

        assert "gui_agent" in TASKS

    def test_parse_bbox(self):
        from flashvlm.tasks.gui_agent import GUIAgentTask

        task = GUIAgentTask(model=None)
        bbox = task._parse_bbox("[0.1, 0.2, 0.8, 0.9]")
        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(0.1)
        assert bbox[3] == pytest.approx(0.9)

    def test_parse_action(self):
        from flashvlm.tasks.gui_agent import GUIAgentTask

        task = GUIAgentTask(model=None)
        action = task._parse_action("click at 0.5 0.3")
        assert action.action_type == "click"

    def test_parse_plan(self):
        from flashvlm.tasks.gui_agent import GUIAgentTask

        task = GUIAgentTask(model=None)
        plan = task._parse_plan("1. Click the search bar\n2. Type query\n3. Press enter")
        assert len(plan) == 3
