"""Model export utilities for deployment (ONNX, TorchScript, SafeTensors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashvlm.cfg.config import FlashVLMConfig


class Exporter:
    """Export FlashVLM models to various deployment formats."""

    def __init__(self, model: nn.Module, config: Optional[FlashVLMConfig] = None):
        self.model = model
        self.config = config or getattr(model, "config", FlashVLMConfig())

    def export(
        self,
        format: str = "safetensors",
        output_path: str = "exported_model",
        quantize: bool = False,
        **kwargs: Any,
    ) -> str:
        """Export model to the specified format.

        Args:
            format: Export format ('onnx', 'torchscript', 'safetensors').
            output_path: Output directory or file path.
            quantize: Whether to apply quantization before export.

        Returns:
            Path to the exported model.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if quantize:
            self._quantize_model()

        if format == "onnx":
            return self._export_onnx(output_path, **kwargs)
        elif format == "torchscript":
            return self._export_torchscript(output_path, **kwargs)
        elif format == "safetensors":
            return self._export_safetensors(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'onnx', 'torchscript', or 'safetensors'.")

    def _export_onnx(self, output_path: Path, **kwargs: Any) -> str:
        """Export model to ONNX format."""
        onnx_path = output_path / "model.onnx"
        self.model.eval()

        img_size = self.config.vision.image_size
        dummy_pixel_values = torch.randn(1, 3, img_size, img_size)
        dummy_input_ids = torch.randint(0, 1000, (1, 32))
        dummy_attention_mask = torch.ones(1, 32, dtype=torch.long)

        try:
            torch.onnx.export(
                self.model,
                (dummy_input_ids, dummy_pixel_values, dummy_attention_mask),
                str(onnx_path),
                input_names=["input_ids", "pixel_values", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "pixel_values": {0: "batch"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
            print(f"ONNX model exported to: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("Saving state dict instead.")
            torch.save(self.model.state_dict(), output_path / "model_state.pt")
            return str(output_path / "model_state.pt")

        return str(onnx_path)

    def _export_torchscript(self, output_path: Path, **kwargs: Any) -> str:
        """Export model to TorchScript format."""
        ts_path = output_path / "model.pt"
        self.model.eval()

        try:
            scripted = torch.jit.script(self.model)
            scripted.save(str(ts_path))
            print(f"TorchScript model exported to: {ts_path}")
        except Exception:
            try:
                img_size = self.config.vision.image_size
                dummy_inputs = (
                    torch.randint(0, 1000, (1, 32)),
                    torch.randn(1, 3, img_size, img_size),
                    torch.ones(1, 32, dtype=torch.long),
                )
                traced = torch.jit.trace(self.model, dummy_inputs)
                traced.save(str(ts_path))
                print(f"TorchScript (traced) model exported to: {ts_path}")
            except Exception as e:
                print(f"TorchScript export failed: {e}")
                torch.save(self.model.state_dict(), ts_path)
                print(f"State dict saved to: {ts_path}")

        return str(ts_path)

    def _export_safetensors(self, output_path: Path, **kwargs: Any) -> str:
        """Export model weights in SafeTensors format."""
        st_path = output_path / "model.safetensors"

        try:
            from safetensors.torch import save_file

            state_dict = {k: v.contiguous() for k, v in self.model.state_dict().items()}
            save_file(state_dict, str(st_path))
            print(f"SafeTensors model exported to: {st_path}")
        except ImportError:
            print("safetensors not installed, falling back to torch.save")
            fallback_path = output_path / "model.pt"
            torch.save(self.model.state_dict(), fallback_path)
            return str(fallback_path)

        self.config.save_yaml(output_path / "config.yaml")
        return str(st_path)

    def _quantize_model(self) -> None:
        """Apply dynamic quantization to the model."""
        self.model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        print("Dynamic quantization applied (INT8).")

    def get_model_size(self) -> Dict[str, Any]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(size_mb, 2),
            "parameter_ratio": f"{trainable_params / total_params * 100:.1f}% trainable",
        }
