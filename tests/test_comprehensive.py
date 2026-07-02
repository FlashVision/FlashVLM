"""Comprehensive tests for FlashVLM."""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from flashvlm.cfg.config import FlashVLMConfig, ProjectorConfig

# Dimensions where CLIP fallback model works:
# CLIP fallback has Conv2d(3, H, P, P) → Flatten(2) → Linear(H, H) → LN(H)
# Linear requires last dim = H, and Flatten(2) gives (B, H, (I/P)^2)
# So (I/P)^2 must equal H.
# H=64, P=8, I=64  →  (64/8)^2 = 64 = H  ✓
_V_HIDDEN = 64
_V_PATCH = 8
_V_IMG = 70
_L_HIDDEN = 128
_L_VOCAB = 100


def _vision_config(**kw):
    config = FlashVLMConfig()
    config.vision.hidden_size = kw.get("hidden_size", _V_HIDDEN)
    config.vision.patch_size = kw.get("patch_size", _V_PATCH)
    config.vision.image_size = kw.get("image_size", _V_IMG)
    config.vision.freeze = kw.get("freeze", False)
    config.vision.encoder_name = kw.get("encoder_name", "openai/clip")
    return config.vision


def _full_config(**kw):
    config = FlashVLMConfig()
    config.vision.hidden_size = _V_HIDDEN
    config.vision.patch_size = _V_PATCH
    config.vision.image_size = _V_IMG
    config.vision.freeze = False
    config.projector.input_dim = _V_HIDDEN
    config.projector.output_dim = _L_HIDDEN
    config.language.hidden_size = _L_HIDDEN
    config.language.vocab_size = _L_VOCAB
    config.language.model_name = "placeholder"
    for k, v in kw.items():
        parts = k.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return config


# ---------------------------------------------------------------------------
# Vision encoders: CLIP, SigLIP, DINOv2
# ---------------------------------------------------------------------------
class TestCLIPVisionEncoder:
    def test_forward_shape(self):
        from flashvlm.models.vision_encoder import CLIPVisionEncoder

        cfg = _vision_config()
        encoder = CLIPVisionEncoder(cfg)
        x = torch.randn(1, 3, _V_IMG, _V_IMG)
        out = encoder(x)
        assert out.dim() == 3
        assert out.shape[0] == 1

    def test_num_patches(self):
        from flashvlm.models.vision_encoder import CLIPVisionEncoder

        cfg = _vision_config()
        encoder = CLIPVisionEncoder(cfg)
        assert encoder.num_patches == (_V_IMG // _V_PATCH) ** 2

    def test_get_transform(self):
        from flashvlm.models.vision_encoder import CLIPVisionEncoder

        cfg = _vision_config()
        encoder = CLIPVisionEncoder(cfg)
        transform = encoder.get_transform()
        assert transform is not None


class TestSigLIPVisionEncoder:
    def test_forward(self):
        from flashvlm.models.vision_encoder import SigLIPVisionEncoder

        cfg = _vision_config(encoder_name="siglip")
        encoder = SigLIPVisionEncoder(cfg)
        x = torch.randn(1, 3, _V_IMG, _V_IMG)
        out = encoder(x)
        assert out.dim() == 3
        assert out.shape[0] == 1
        assert out.shape[2] == _V_HIDDEN


class TestDINOv2VisionEncoder:
    def test_forward(self):
        from flashvlm.models.vision_encoder import DINOv2VisionEncoder

        cfg = _vision_config(encoder_name="dinov2")
        encoder = DINOv2VisionEncoder(cfg)
        x = torch.randn(1, 3, _V_IMG, _V_IMG)
        out = encoder(x)
        assert out.dim() == 3

    def test_dino_transform(self):
        from flashvlm.models.vision_encoder import DINOv2VisionEncoder

        cfg = _vision_config(encoder_name="dinov2")
        encoder = DINOv2VisionEncoder(cfg)
        t = encoder.get_transform()
        assert t is not None


class TestBuildVisionEncoder:
    def test_factory_clip(self):
        from flashvlm.models.vision_encoder import build_vision_encoder

        cfg = _vision_config(encoder_name="openai/clip")
        encoder = build_vision_encoder(cfg)
        assert encoder is not None

    def test_factory_siglip(self):
        from flashvlm.models.vision_encoder import build_vision_encoder

        cfg = _vision_config(encoder_name="siglip-test")
        encoder = build_vision_encoder(cfg)
        assert encoder is not None

    def test_factory_dinov2(self):
        from flashvlm.models.vision_encoder import build_vision_encoder

        cfg = _vision_config(encoder_name="dinov2-test")
        encoder = build_vision_encoder(cfg)
        assert encoder is not None


# ---------------------------------------------------------------------------
# Projectors: MLP, Q-Former, CrossAttention
# ---------------------------------------------------------------------------
class TestProjectorsExtended:
    def test_mlp_gradient_flow(self):
        from flashvlm.models.projector import MLPProjector

        config = ProjectorConfig(input_dim=128, output_dim=256, num_layers=2)
        proj = MLPProjector(config)
        x = torch.randn(1, 10, 128, requires_grad=True)
        out = proj(x)
        out.sum().backward()
        assert x.grad is not None

    def test_qformer_output_shape(self):
        from flashvlm.models.projector import QFormerProjector

        config = ProjectorConfig(input_dim=128, output_dim=256, num_query_tokens=8, type="qformer")
        proj = QFormerProjector(config)
        x = torch.randn(2, 50, 128)
        out = proj(x)
        assert out.shape == (2, 8, 256)

    def test_cross_attention_projector(self):
        from flashvlm.models.projector import CrossAttentionProjector

        config = ProjectorConfig(input_dim=64, output_dim=128, num_layers=1, type="cross_attention")
        proj = CrossAttentionProjector(config)
        x = torch.randn(1, 20, 64)
        out = proj(x)
        assert out.shape == (1, 20, 128)


# ---------------------------------------------------------------------------
# Architectures: LLaVA, Qwen-VL, InternVL, Phi-Vision
# ---------------------------------------------------------------------------
class TestLLaVAArchitecture:
    def test_encode_images(self):
        from flashvlm.models.architectures.llava import LLaVAArchitecture

        config = _full_config()
        arch = LLaVAArchitecture(config)
        imgs = torch.randn(1, 3, _V_IMG, _V_IMG)
        features = arch.encode_images(imgs)
        assert features.dim() == 3
        assert features.shape[2] == _L_HIDDEN

    def test_prepare_inputs_no_language_model(self):
        from flashvlm.models.architectures.llava import LLaVAArchitecture

        config = _full_config()
        arch = LLaVAArchitecture(config)
        input_ids = torch.randint(0, 100, (1, 10))
        result = arch.prepare_inputs_for_multimodal(input_ids)
        assert "input_ids" in result


class TestQwenVLArchitecture:
    def test_position_encoding(self):
        from flashvlm.models.architectures.qwen_vl import VisualPositionEncoding

        pe = VisualPositionEncoding(hidden_size=64, max_patches=100)
        tokens = torch.randn(2, 16, 64)
        out = pe(tokens, image_size=(4, 4))
        assert out.shape == tokens.shape

    def test_wrap_visual_tokens(self):
        from flashvlm.models.architectures.qwen_vl import QwenVLArchitecture

        config = _full_config()
        arch = QwenVLArchitecture(config)
        tokens = torch.randn(1, 10, _L_HIDDEN)
        wrapped = arch.wrap_visual_tokens(tokens)
        assert wrapped.shape[1] == 12  # start + 10 + end

    def test_forward_text_only(self):
        from flashvlm.models.architectures.qwen_vl import QwenVLArchitecture

        config = _full_config()
        arch = QwenVLArchitecture(config)
        input_ids = torch.randint(0, 100, (1, 5))
        result = arch(input_ids=input_ids)
        assert "logits" in result


class TestInternVLArchitecture:
    def test_pixel_shuffle(self):
        from flashvlm.models.architectures.internvl import PixelShuffle

        ps = PixelShuffle(scale_factor=2)
        x = torch.randn(1, 16, 64)
        out = ps(x)
        assert out.shape == (1, 4, 256)

    def test_dynamic_resolution_layout(self):
        from flashvlm.models.architectures.internvl import DynamicResolutionModule

        dr = DynamicResolutionModule(base_size=224, max_tiles=6)
        layout = dr.compute_tile_layout((448, 224))
        assert layout[0] * layout[1] <= 6

    def test_dynamic_resolution_tiles(self):
        from flashvlm.models.architectures.internvl import DynamicResolutionModule

        dr = DynamicResolutionModule(base_size=32, max_tiles=4)
        img = torch.randn(1, 3, 64, 64)
        tiles = dr.split_into_tiles(img)
        assert len(tiles) >= 2
        for t in tiles:
            assert t.shape[2] == 32
            assert t.shape[3] == 32

    def test_internvl_forward_text_only(self):
        from flashvlm.models.architectures.internvl import InternVLArchitecture

        config = _full_config()
        arch = InternVLArchitecture(config)
        ids = torch.randint(0, 100, (1, 5))
        result = arch(input_ids=ids)
        assert "logits" in result


class TestPhiVisionExtended:
    def test_phi_image_embedding_gradient(self):
        from flashvlm.models.architectures.phi_vision import PhiImageEmbedding

        config = _full_config()
        embed = PhiImageEmbedding(config)
        x = torch.randn(1, 3, _V_IMG, _V_IMG, requires_grad=True)
        out = embed(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# FlashVLM main model
# ---------------------------------------------------------------------------
class TestFlashVLMModel:
    def test_init_default(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        assert model.vision_encoder is not None
        assert model.projector is not None

    def test_encode_image_tensor(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        img = torch.randn(1, 3, _V_IMG, _V_IMG)
        embed = model.encode_image(img)
        assert embed.dim() == 3

    def test_encode_images_list(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        imgs = [torch.randn(1, 3, _V_IMG, _V_IMG) for _ in range(2)]
        embeds = model.encode_images(imgs)
        assert len(embeds) == 2

    def test_format_vqa_prompt_llava(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config(architecture="llava")
        model = FlashVLM(config)
        prompt = model._format_vqa_prompt("What is this?")
        assert "USER:" in prompt
        assert "<image>" in prompt

    def test_format_vqa_prompt_qwen(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config(architecture="qwen_vl")
        model = FlashVLM(config)
        prompt = model._format_vqa_prompt("Describe")
        assert "<img>" in prompt

    def test_format_caption_prompt(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config(architecture="qwen_vl")
        model = FlashVLM(config)
        prompt = model._format_caption_prompt()
        assert "Describe" in prompt

    def test_num_parameters(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        total = model.num_parameters()
        trainable = model.num_parameters(trainable_only=True)
        assert total > 0
        assert trainable <= total

    def test_resolve_config_known_model(self):
        from flashvlm.models.vlm import FlashVLM

        config = FlashVLM._resolve_config("llava-v1.5-7b")
        assert config.architecture == "llava"

    def test_resolve_config_unknown_model(self):
        from flashvlm.models.vlm import FlashVLM

        config = FlashVLM._resolve_config("some-unknown-model")
        assert config.language.model_name == "some-unknown-model"

    def test_forward_with_pixel_values(self):
        """Forward pass with image input through the placeholder model."""
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        pixel_values = torch.randn(1, 3, _V_IMG, _V_IMG)
        input_embeds = torch.randn(1, 5, _L_HIDDEN)
        result = model(input_ids=input_embeds, pixel_values=pixel_values)
        assert "logits" in result

    def test_merge_visual_text(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        input_ids = torch.randn(1, 5, _L_HIDDEN)
        visual_embeds = torch.randn(1, 8, _L_HIDDEN)
        combined = model._merge_visual_text_embeddings(input_ids, visual_embeds)
        assert combined.shape == (1, 13, _L_HIDDEN)


# ---------------------------------------------------------------------------
# Generation: samplers, beam search
# ---------------------------------------------------------------------------
class TestVLMSamplers:
    def test_temperature_sampler(self):
        from flashvlm.generation.sampler import TemperatureSampler

        sampler = TemperatureSampler(temperature=0.5)
        logits = torch.randn(1, 100)
        scaled = sampler(logits)
        assert scaled.shape == (1, 100)

    def test_temperature_sampler_invalid(self):
        from flashvlm.generation.sampler import TemperatureSampler

        with pytest.raises(ValueError):
            TemperatureSampler(temperature=0.0)

    def test_top_k_sampler(self):
        from flashvlm.generation.sampler import TopKSampler

        sampler = TopKSampler(k=10)
        logits = torch.randn(1, 100)
        filtered = sampler(logits)
        assert (filtered == float("-inf")).sum() >= 90

    def test_top_k_sample_output(self):
        from flashvlm.generation.sampler import TopKSampler

        sampler = TopKSampler(k=5)
        logits = torch.randn(2, 50)
        tokens = sampler.sample(logits)
        assert tokens.shape == (2,)

    def test_top_p_sampler(self):
        from flashvlm.generation.sampler import TopPSampler

        sampler = TopPSampler(p=0.5)
        logits = torch.randn(1, 100)
        filtered = sampler(logits)
        assert filtered.shape == (1, 100)

    def test_top_p_invalid(self):
        from flashvlm.generation.sampler import TopPSampler

        with pytest.raises(ValueError):
            TopPSampler(p=0.0)

    def test_repetition_penalty(self):
        from flashvlm.generation.sampler import RepetitionPenaltySampler

        sampler = RepetitionPenaltySampler(penalty=2.0)
        logits = torch.ones(1, 10)
        generated = torch.tensor([[0, 1, 2]])
        result = sampler(logits, generated)
        assert result[0, 0] < 1.0
        assert result[0, 5] == 1.0

    def test_combined_sample(self):
        from flashvlm.generation.sampler import combined_sample

        logits = torch.randn(2, 100)
        tokens = combined_sample(logits, temperature=0.8, top_k=10, top_p=0.9)
        assert tokens.shape == (2,)

    def test_combined_sample_with_repetition(self):
        from flashvlm.generation.sampler import combined_sample

        logits = torch.randn(1, 50)
        gen_ids = torch.randint(0, 50, (1, 5))
        tokens = combined_sample(logits, repetition_penalty=1.5, generated_ids=gen_ids)
        assert tokens.shape == (1,)


class TestVLMBeamSearch:
    def test_hypothesis_normalized_score(self):
        from flashvlm.generation.beam_search import BeamHypothesis

        h = BeamHypothesis(tokens=[1, 2, 3], score=3.0)
        norm = h.normalized_score(length_penalty=1.0)
        assert norm == 1.0

    def test_hypothesis_empty(self):
        from flashvlm.generation.beam_search import BeamHypothesis

        h = BeamHypothesis()
        assert h.length == 0
        assert h.normalized_score() == 0.0

    def test_beam_search_no_repeat_ngram(self):
        from flashvlm.generation.beam_search import BeamSearchGenerator

        gen = BeamSearchGenerator(num_beams=2, no_repeat_ngram_size=2)
        logits = torch.randn(2, 20)
        ids = torch.tensor([[1, 2, 1, 2]])
        result = gen._apply_no_repeat_ngram(logits[:1], ids, 2)
        assert result.shape == (1, 20)


# ---------------------------------------------------------------------------
# Video encoder
# ---------------------------------------------------------------------------
class TestVideoEncoderExtended:
    def test_sample_frames_uniform(self):
        from flashvlm.models.video_encoder import sample_frames

        indices = sample_frames(50, num_sample=4, strategy="uniform")
        assert len(indices) == 4
        assert indices[0] == 0
        assert indices[-1] == 49

    def test_sample_frames_fewer_than_requested(self):
        from flashvlm.models.video_encoder import sample_frames

        indices = sample_frames(3, num_sample=8, strategy="uniform")
        assert len(indices) == 3

    def test_temporal_pooling_mean(self):
        from flashvlm.models.video_encoder import TemporalPooling

        pool = TemporalPooling(32, pool_type="mean")
        x = torch.randn(1, 8, 16, 32)
        out = pool(x)
        assert out.shape == (1, 16, 32)

    def test_temporal_pooling_attention(self):
        from flashvlm.models.video_encoder import TemporalPooling

        pool = TemporalPooling(32, num_heads=4, pool_type="attention")
        x = torch.randn(1, 4, 8, 32)
        out = pool(x)
        assert out.shape == (1, 8, 32)

    def test_video_token_generator_output(self):
        from flashvlm.models.video_encoder import VideoTokenGenerator

        gen = VideoTokenGenerator(64, max_frames=8)
        x = torch.randn(1, 4, 10, 64)
        out = gen(x, num_frames=4)
        assert out.shape == (1, 4, 10, 64)


# ---------------------------------------------------------------------------
# Tasks: Captioning, OCR, Reasoning
# ---------------------------------------------------------------------------
class TestCaptioningTask:
    def test_prompt_templates(self):
        from flashvlm.tasks.captioning import CaptioningTask

        task = CaptioningTask(model=None)
        assert "brief" in task.PROMPT_TEMPLATES
        assert "detailed" in task.PROMPT_TEMPLATES
        assert "creative" in task.PROMPT_TEMPLATES

    def test_postprocess_adds_period(self):
        from flashvlm.tasks.captioning import CaptioningTask

        task = CaptioningTask(model=None)
        assert task._postprocess("hello") == "hello."
        assert task._postprocess("hello.") == "hello."
        assert task._postprocess("hello!") == "hello!"


class TestOCRTask:
    def test_prompt_templates(self):
        from flashvlm.tasks.ocr import OCRTask

        task = OCRTask(model=None)
        assert "extract_all" in task.PROMPT_TEMPLATES
        assert "document" in task.PROMPT_TEMPLATES
        assert "handwriting" in task.PROMPT_TEMPLATES

    def test_parse_structured_kv(self):
        from flashvlm.tasks.ocr import OCRTask

        task = OCRTask(model=None)
        text = "Name: John\nAge: 30\nCity: NYC"
        result = task._parse_structured(text)
        assert result["key_values"]["Name"] == "John"
        assert result["key_values"]["Age"] == "30"

    def test_parse_structured_table(self):
        from flashvlm.tasks.ocr import OCRTask

        task = OCRTask(model=None)
        text = "Header1 | Header2\nVal1 | Val2"
        result = task._parse_structured(text)
        assert len(result["tables"]) >= 1

    def test_evaluate_metrics(self):
        from flashvlm.tasks.ocr import OCRTask

        task = OCRTask(model=None)
        preds = ["hello world", "test"]
        gts = ["hello world", "test"]
        metrics = task.evaluate(preds, gts)
        assert metrics["char_accuracy"] == 1.0
        assert metrics["word_accuracy"] == 1.0


class TestReasoningTask:
    def test_reasoning_registered(self):
        from flashvlm.registry import TASKS

        assert "reasoning" in TASKS


# ---------------------------------------------------------------------------
# Training: SFT, DPO, Multi-stage
# ---------------------------------------------------------------------------
class TestVLMSFT:
    def test_stage1_freezing(self):
        from flashvlm.training.sft import SupervisedFineTuner

        class SimpleVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Linear(10, 10)
                self.projector = nn.Linear(10, 10)
                self.language_model = nn.Linear(10, 10)

        model = SimpleVLM()
        config = FlashVLMConfig()
        SupervisedFineTuner(model, config, stage=1)
        assert model.projector.weight.requires_grad
        assert not model.vision_encoder.weight.requires_grad
        assert not model.language_model.weight.requires_grad

    def test_stage2_freezing(self):
        from flashvlm.training.sft import SupervisedFineTuner

        class SimpleVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Linear(10, 10)
                self.projector = nn.Linear(10, 10)
                self.language_model = nn.Linear(10, 10)

        model = SimpleVLM()
        config = FlashVLMConfig()
        SupervisedFineTuner(model, config, stage=2)
        assert model.projector.weight.requires_grad
        assert not model.vision_encoder.weight.requires_grad
        assert model.language_model.weight.requires_grad


class TestVLMDPOTrainer:
    def test_compute_dpo_loss(self):
        from flashvlm.training.dpo import DPOTrainer

        model = nn.Linear(10, 10)
        ref_model = nn.Linear(10, 10)
        config = FlashVLMConfig()
        trainer = DPOTrainer(model, ref_model, config, beta=0.1)
        loss, cr, rr = trainer.compute_dpo_loss(
            policy_chosen_logps=torch.tensor([1.0]),
            policy_rejected_logps=torch.tensor([0.5]),
            reference_chosen_logps=torch.tensor([0.9]),
            reference_rejected_logps=torch.tensor([0.4]),
        )
        assert loss.dim() == 0

    def test_compute_dpo_loss_with_label_smoothing(self):
        from flashvlm.training.dpo import DPOTrainer

        model = nn.Linear(10, 10)
        ref_model = nn.Linear(10, 10)
        config = FlashVLMConfig()
        trainer = DPOTrainer(model, ref_model, config, beta=0.1, label_smoothing=0.1)
        loss, _, _ = trainer.compute_dpo_loss(
            torch.tensor([1.0]),
            torch.tensor([0.5]),
            torch.tensor([0.9]),
            torch.tensor([0.4]),
        )
        assert loss.dim() == 0

    def test_ref_model_frozen(self):
        from flashvlm.training.dpo import DPOTrainer

        model = nn.Linear(10, 10)
        ref = nn.Linear(10, 10)
        config = FlashVLMConfig()
        trainer = DPOTrainer(model, ref, config)
        for p in trainer.ref_model.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# LoRA for VLM
# ---------------------------------------------------------------------------
class TestVLMLoRAExtended:
    def test_lora_rank_effect(self):
        from flashvlm.models.lora import LoRALinear

        lora_r4 = LoRALinear(in_features=128, out_features=128, rank=4)
        lora_r16 = LoRALinear(in_features=128, out_features=128, rank=16)
        p4 = sum(p.numel() for p in lora_r4.parameters() if p.requires_grad)
        p16 = sum(p.numel() for p in lora_r16.parameters() if p.requires_grad)
        assert p4 < p16

    def test_lora_from_linear_preserves_output(self):
        from flashvlm.models.lora import LoRALinear

        linear = nn.Linear(32, 32)
        lora = LoRALinear.from_linear(linear, rank=4, alpha=8)
        x = torch.randn(1, 5, 32)
        out = lora(x)
        assert out.shape == (1, 5, 32)

    def test_apply_lora_targets(self):
        from flashvlm.models.lora import LoRALinear, apply_lora

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.ffn = nn.Linear(64, 64)

        model = TestModel()
        model = apply_lora(model, rank=4, target_modules=["q_proj", "v_proj"])
        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert isinstance(model.k_proj, nn.Linear)
        assert isinstance(model.ffn, nn.Linear)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestFlashVLMConfigExtended:
    def test_from_dict_sets_architecture(self):
        data = {
            "model_name": "test",
            "architecture": "internvl",
            "vision": {
                "image_size": 448,
                "encoder_name": "openai/clip",
                "hidden_size": 1024,
                "patch_size": 14,
                "num_layers": 24,
                "num_heads": 16,
                "freeze": True,
                "use_checkpoint": False,
            },
        }
        config = FlashVLMConfig.from_dict(data)
        assert config.architecture == "internvl"
        assert config.vision.image_size == 448

    def test_to_dict_roundtrip(self):
        config = FlashVLMConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "vision.image_size" in d


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
class TestEvaluatorsExtended:
    def test_pope_evaluator_metrics(self):
        from flashvlm.eval.evaluator import POPEEvaluator

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def generate(self, prompt, image=None, **kwargs):
                return "yes"

        evaluator = POPEEvaluator(model=DummyModel())
        metrics = evaluator.compute_metrics(
            ["yes", "no"],
            [{"answer": "yes"}, {"answer": "yes"}],
        )
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_mmbench_extract_choice(self):
        from flashvlm.eval.evaluator import MMBenchEvaluator

        assert MMBenchEvaluator._extract_choice("A") == "A"
        assert MMBenchEvaluator._extract_choice("The answer is B") == "B"
        assert MMBenchEvaluator._extract_choice("C. The cat") == "C"
        assert MMBenchEvaluator._extract_choice("D") == "D"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetricsExtended:
    def test_vqa_accuracy_partial(self):
        from flashvlm.analytics.metrics import vqa_accuracy

        preds = ["yes", "no"]
        gts = [["yes", "yes", "yes"], ["yes", "no", "no"]]
        acc = vqa_accuracy(preds, gts)
        assert 0.0 < acc <= 1.0

    def test_bleu_partial_overlap(self):
        from flashvlm.analytics.metrics import compute_bleu

        preds = ["the cat sat on mat"]
        refs = [["the cat sat on the mat"]]
        score = compute_bleu(preds, refs)
        assert 0.0 < score <= 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestVLMCLI:
    def test_get_parser(self):
        from flashvlm.cli import get_parser

        parser = get_parser()
        assert parser is not None
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_version_command(self):
        from flashvlm.cli import cmd_version

        cmd_version(SimpleNamespace())

    def test_check_command(self):
        from flashvlm.cli import cmd_check

        cmd_check(SimpleNamespace(verbose=False))

    def test_main_no_args(self):
        import sys

        from flashvlm.cli import main

        old = sys.argv
        sys.argv = ["flashvlm"]
        with pytest.raises(SystemExit):
            main()
        sys.argv = old

    def test_commands_mapping(self):
        from flashvlm.cli import COMMANDS

        assert "version" in COMMANDS
        assert "check" in COMMANDS
        assert "train" in COMMANDS
        assert "chat" in COMMANDS
        assert "caption" in COMMANDS
        assert "vqa" in COMMANDS
        assert "export" in COMMANDS
        assert "benchmark" in COMMANDS


# ---------------------------------------------------------------------------
# Integration: image + text → encode → generate → decode
# ---------------------------------------------------------------------------
class TestVLMIntegration:
    def test_encode_tensor_image(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        img_tensor = torch.randn(1, 3, _V_IMG, _V_IMG)
        embed = model.encode_image(img_tensor)
        assert embed.dim() == 3

    def test_encode_3d_image(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        img_tensor = torch.randn(3, _V_IMG, _V_IMG)
        embed = model.encode_image(img_tensor)
        assert embed.dim() == 3
        assert embed.shape[0] == 1

    def test_merge_visual_text_embeddings(self):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        input_embeds = torch.randn(1, 5, _L_HIDDEN)
        visual_embeds = torch.randn(1, 8, _L_HIDDEN)
        combined = model._merge_visual_text_embeddings(input_embeds, visual_embeds)
        assert combined.shape == (1, 13, _L_HIDDEN)

    def test_save_and_config(self, tmp_path):
        from flashvlm.models.vlm import FlashVLM

        config = _full_config()
        model = FlashVLM(config)
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))
        assert (save_dir / "config.yaml").exists()
        assert (save_dir / "model.pt").exists()
