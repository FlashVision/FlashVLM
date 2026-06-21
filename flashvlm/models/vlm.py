"""Main FlashVLM model class — unified Vision-Language Model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from flashvlm.cfg.config import FlashVLMConfig
from flashvlm.models.projector import Projector, build_projector
from flashvlm.models.vision_encoder import VisionEncoder, build_vision_encoder
from flashvlm.registry import MODELS


@MODELS.register("flashvlm")
class FlashVLM(nn.Module):
    """FlashVLM: Unified Vision-Language Model.

    Combines a vision encoder, a vision-to-language projector, and a
    language model decoder for multimodal understanding and generation.
    """

    def __init__(self, config: FlashVLMConfig | None = None):
        super().__init__()
        self.config = config or FlashVLMConfig()
        self._device = torch.device("cpu")
        self._dtype = torch.float32

        self.vision_encoder: VisionEncoder | None = None
        self.projector: Projector | None = None
        self.language_model: nn.Module | None = None
        self.tokenizer = None
        self.image_processor = None

        self._build_model()

    def _build_model(self) -> None:
        """Initialize model components based on configuration."""
        self.vision_encoder = build_vision_encoder(self.config.vision)
        self.projector = build_projector(self.config.projector)
        self._init_language_model()

    def _init_language_model(self) -> None:
        """Load the language model backbone from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = self.config.language.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.language_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            if self.config.language.freeze:
                for param in self.language_model.parameters():
                    param.requires_grad = False

        except Exception as e:
            model_name = self.config.language.model_name
            print(f"Warning: Could not load language model '{model_name}': {e}")
            print("Using a placeholder embedding layer.")
            self.language_model = nn.Sequential(
                nn.Linear(self.config.language.hidden_size, self.config.language.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.language.hidden_size, self.config.language.vocab_size),
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> FlashVLM:
        """Load a pretrained FlashVLM model.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            device: Target device ('auto', 'cuda', 'cpu').
            dtype: Model dtype (None for auto-detection).

        Returns:
            Initialized FlashVLM model.
        """
        config = cls._resolve_config(model_name_or_path)
        model = cls(config)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        model._device = torch.device(device)
        model._dtype = dtype
        model = model.to(device=model._device, dtype=model._dtype)

        model.eval()
        return model

    @staticmethod
    def _resolve_config(model_name_or_path: str) -> FlashVLMConfig:
        """Resolve model name/path to a FlashVLMConfig."""
        path = Path(model_name_or_path)
        if path.exists() and (path / "config.yaml").exists():
            return FlashVLMConfig.from_yaml(path / "config.yaml")

        model_configs = {
            "llava-v1.5-7b": {
                "architecture": "llava",
                "model_name": "llava-v1.5-7b",
                "language": {"model_name": "meta-llama/Llama-2-7b-hf"},
                "vision": {"encoder_name": "openai/clip-vit-large-patch14-336"},
            },
            "qwen-vl-7b": {
                "architecture": "qwen_vl",
                "model_name": "qwen-vl-7b",
                "language": {"model_name": "Qwen/Qwen-VL-Chat"},
                "vision": {"encoder_name": "openai/clip-vit-large-patch14-336"},
            },
            "internvl-7b": {
                "architecture": "internvl",
                "model_name": "internvl-7b",
                "language": {"model_name": "internlm/internlm2-chat-7b"},
                "vision": {"encoder_name": "OpenGVLab/InternViT-6B-448px-V1-5"},
            },
            "phi-vision-4b": {
                "architecture": "phi_vision",
                "model_name": "phi-vision-4b",
                "language": {"model_name": "microsoft/phi-2", "hidden_size": 2560},
                "vision": {"encoder_name": "google/siglip-so400m-patch14-384"},
            },
        }

        if model_name_or_path in model_configs:
            return FlashVLMConfig.from_dict(model_configs[model_name_or_path])

        config = FlashVLMConfig()
        config.language.model_name = model_name_or_path
        return config

    def encode_image(self, image: str | Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode an image into visual embeddings.

        Args:
            image: Image as PIL Image, file path, or preprocessed tensor.

        Returns:
            Visual embeddings tensor of shape (1, num_tokens, hidden_dim).
        """
        pixel_values = self._preprocess_image(image)
        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)

        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values)
            projected = self.projector(vision_features)

        return projected

    def encode_images(
        self, images: list[str | Path | Image.Image | torch.Tensor]
    ) -> list[torch.Tensor]:
        """Encode multiple images into visual embeddings.

        Args:
            images: List of images (PIL, path, or tensor).

        Returns:
            List of visual embedding tensors, each (1, num_tokens, hidden_dim).
        """
        return [self.encode_image(img) for img in images]

    def _preprocess_image(self, image: str | Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Convert various image inputs to a preprocessed tensor."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        transform = self.vision_encoder.get_transform()
        pixel_values = transform(image).unsqueeze(0)
        return pixel_values

    def _interleave_multi_image_tokens(
        self,
        input_ids: torch.Tensor,
        image_embeds_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Replace multiple <image> tokens in the input with corresponding visual embeddings.

        Supports prompts like: "<image_1> What is in image 1? <image_2> Compare with image 2."
        Falls back to concatenating all image embeddings before text if no placeholders found.
        """
        if hasattr(self.language_model, "get_input_embeddings"):
            embed_layer = self.language_model.get_input_embeddings()
        else:
            embed_layer = self.language_model[0]

        text_embeds = embed_layer(input_ids)
        image_token_index = -200

        image_positions = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]

        if len(image_positions) == 0:
            all_visual = torch.cat(image_embeds_list, dim=1)
            return torch.cat([all_visual, text_embeds], dim=1)

        segments = []
        prev_pos = 0
        img_idx = 0

        for pos in image_positions:
            pos = pos.item()
            if prev_pos < pos:
                segments.append(text_embeds[:, prev_pos:pos, :])
            if img_idx < len(image_embeds_list):
                segments.append(image_embeds_list[img_idx])
                img_idx += 1
            prev_pos = pos + 1

        if prev_pos < text_embeds.shape[1]:
            segments.append(text_embeds[:, prev_pos:, :])

        while img_idx < len(image_embeds_list):
            segments.append(image_embeds_list[img_idx])
            img_idx += 1

        return torch.cat(segments, dim=1)

    def generate(
        self,
        prompt: str,
        image: str | Path | Image.Image | torch.Tensor | None = None,
        images: list[str | Path | Image.Image | torch.Tensor] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text given a prompt and optional image(s).

        Args:
            prompt: Text prompt for generation.
            image: Optional single image input.
            images: Optional list of images for multi-image conversations.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.

        Returns:
            Generated text string.
        """
        gen_config = self.config.generation
        max_new_tokens = max_new_tokens or gen_config.max_new_tokens
        temperature = temperature or gen_config.temperature
        top_p = top_p or gen_config.top_p
        top_k = top_k or gen_config.top_k

        if self.tokenizer is None:
            return "[Model tokenizer not loaded. Please load a valid language model.]"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        attention_mask = torch.ones_like(input_ids)

        has_images = image is not None or (images is not None and len(images) > 0)

        if has_images:
            if images is not None and len(images) > 0:
                image_embeds_list = self.encode_images(images)
                inputs_embeds = self._interleave_multi_image_tokens(input_ids, image_embeds_list)
            else:
                visual_embeds = self.encode_image(image)
                inputs_embeds = self._merge_visual_text_embeddings(input_ids, visual_embeds)
            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
        else:
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

        gen_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.language_model.generate(**gen_kwargs)

        generated = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _merge_visual_text_embeddings(
        self, input_ids: torch.Tensor, visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Merge visual and text embeddings for the language model."""
        if hasattr(self.language_model, "get_input_embeddings"):
            embed_layer = self.language_model.get_input_embeddings()
        else:
            embed_layer = self.language_model[0]

        text_embeds = embed_layer(input_ids)
        combined = torch.cat([visual_embeds, text_embeds], dim=1)
        return combined

    def ask(
        self,
        question: str,
        image: str | Path | Image.Image | torch.Tensor | None = None,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> str:
        """Ask a question about an image (convenience method for VQA).

        Args:
            question: The question to ask.
            image: Image to ask about.
            max_new_tokens: Max generation length.

        Returns:
            The model's answer as a string.
        """
        prompt = self._format_vqa_prompt(question)
        return self.generate(prompt=prompt, image=image, max_new_tokens=max_new_tokens, **kwargs)

    def caption(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        max_new_tokens: int = 100,
        **kwargs: Any,
    ) -> str:
        """Generate a caption for an image.

        Args:
            image: Image to caption.
            max_new_tokens: Max generation length.

        Returns:
            Generated caption string.
        """
        prompt = self._format_caption_prompt()
        return self.generate(prompt=prompt, image=image, max_new_tokens=max_new_tokens, **kwargs)

    def _format_vqa_prompt(self, question: str) -> str:
        """Format a VQA prompt based on model architecture."""
        arch = self.config.architecture
        if arch == "llava":
            return f"USER: <image>\n{question}\nASSISTANT:"
        elif arch == "qwen_vl":
            return f"<img></img>\n{question}"
        elif arch == "internvl":
            return f"<image>\n{question}"
        else:
            return f"Question: {question}\nAnswer:"

    def _format_caption_prompt(self) -> str:
        """Format a captioning prompt based on model architecture."""
        arch = self.config.architecture
        if arch == "llava":
            return "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        elif arch == "qwen_vl":
            return "<img></img>\nDescribe this image."
        else:
            return "Describe this image in detail."

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            input_ids: Tokenized text input (B, seq_len).
            pixel_values: Preprocessed images (B, 3, H, W).
            attention_mask: Attention mask (B, seq_len).
            labels: Target token IDs for loss computation.

        Returns:
            Dictionary with 'loss' and 'logits' tensors.
        """
        if pixel_values is not None:
            visual_features = self.vision_encoder(pixel_values)
            visual_embeds = self.projector(visual_features)
        else:
            visual_embeds = None

        if hasattr(self.language_model, "get_input_embeddings"):
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
        else:
            text_embeds = self.language_model[0](input_ids)

        if visual_embeds is not None:
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            if attention_mask is not None:
                visual_mask = torch.ones(
                    visual_embeds.shape[:2],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
            if labels is not None:
                visual_labels = torch.full(
                    visual_embeds.shape[:2], -100, device=labels.device, dtype=labels.dtype
                )
                labels = torch.cat([visual_labels, labels], dim=1)
        else:
            inputs_embeds = text_embeds

        if hasattr(self.language_model, "forward"):
            try:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                return {"loss": outputs.loss, "logits": outputs.logits}
            except (TypeError, AttributeError):
                pass

        logits = (
            self.language_model(inputs_embeds)
            if isinstance(self.language_model, nn.Sequential)
            else inputs_embeds
        )

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, path: str | Path) -> None:
        """Save model weights and config to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_yaml(path / "config.yaml")
        torch.save(self.state_dict(), path / "model.pt")

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
