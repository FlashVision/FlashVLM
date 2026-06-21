"""Video encoder for FlashVLM: frame sampling, temporal pooling, video tokens."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvlm.cfg.config import VisionConfig
from flashvlm.models.vision_encoder import VisionEncoder, build_vision_encoder
from flashvlm.registry import VISION_ENCODERS


class TemporalPooling(nn.Module):
    """Temporal pooling over frame-level features using attention or averaging."""

    def __init__(self, hidden_size: int, num_heads: int = 8, pool_type: str = "attention"):
        super().__init__()
        self.pool_type = pool_type
        self.hidden_size = hidden_size

        if pool_type == "attention":
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads,
                dropout=0.1, batch_first=True,
            )
            self.temporal_norm = nn.LayerNorm(hidden_size)
            self.temporal_ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 4, hidden_size),
            )
            self.ffn_norm = nn.LayerNorm(hidden_size)
        elif pool_type == "conv":
            self.temporal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
            self.temporal_norm = nn.LayerNorm(hidden_size)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Pool temporal features across frames.

        Args:
            frame_features: (B, num_frames, num_tokens, hidden) per-frame visual features.

        Returns:
            Pooled features (B, num_output_tokens, hidden).
        """
        B, T, N, D = frame_features.shape

        if self.pool_type == "mean":
            return frame_features.mean(dim=1)

        if self.pool_type == "attention":
            x = frame_features.reshape(B, T * N, D)
            normed = self.temporal_norm(x)
            attn_out, _ = self.temporal_attn(normed, normed, normed)
            x = x + attn_out
            ffn_out = self.temporal_ffn(self.ffn_norm(x))
            x = x + ffn_out
            x = x.reshape(B, T, N, D).mean(dim=1)
            return x

        if self.pool_type == "conv":
            x = frame_features.reshape(B * N, T, D)
            x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.temporal_norm(x.mean(dim=1))
            return x.reshape(B, N, D)

        return frame_features.mean(dim=1)


class VideoTokenGenerator(nn.Module):
    """Generate video tokens with temporal position embeddings."""

    def __init__(self, hidden_size: int, max_frames: int = 64):
        super().__init__()
        self.temporal_embed = nn.Embedding(max_frames, hidden_size)
        self.frame_proj = nn.Linear(hidden_size, hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, frame_features: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Add temporal position encoding to frame features.

        Args:
            frame_features: (B, num_frames, num_tokens, hidden).
            num_frames: Actual number of frames.

        Returns:
            Temporally-encoded features (B, num_frames, num_tokens, hidden).
        """
        B, T, N, D = frame_features.shape
        frame_ids = torch.arange(T, device=frame_features.device)
        temporal_emb = self.temporal_embed(frame_ids)
        temporal_emb = temporal_emb.unsqueeze(0).unsqueeze(2)
        x = frame_features + temporal_emb
        x = self.frame_proj(x)
        x = self.token_norm(x)
        return x


def sample_frames(
    total_frames: int,
    num_sample: int = 8,
    strategy: str = "uniform",
    fps: Optional[float] = None,
    target_fps: float = 1.0,
) -> list[int]:
    """Sample frame indices from a video.

    Args:
        total_frames: Total number of frames in the video.
        num_sample: Desired number of frames to sample.
        strategy: Sampling strategy ('uniform', 'fps', 'keyframe').
        fps: Video FPS (required for 'fps' strategy).
        target_fps: Target FPS for 'fps' strategy.

    Returns:
        List of sampled frame indices.
    """
    if total_frames <= num_sample:
        return list(range(total_frames))

    if strategy == "uniform":
        indices = torch.linspace(0, total_frames - 1, num_sample).long().tolist()
        return indices

    if strategy == "fps" and fps is not None:
        frame_interval = max(1, int(fps / target_fps))
        indices = list(range(0, total_frames, frame_interval))[:num_sample]
        if len(indices) < num_sample:
            remaining = num_sample - len(indices)
            extra = torch.linspace(0, total_frames - 1, remaining).long().tolist()
            indices = sorted(set(indices + extra))[:num_sample]
        return indices

    return torch.linspace(0, total_frames - 1, num_sample).long().tolist()


def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    strategy: str = "uniform",
    image_size: int = 336,
) -> Tuple[torch.Tensor, dict]:
    """Load and preprocess video frames from a file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        strategy: Frame sampling strategy.
        image_size: Target image size for each frame.

    Returns:
        Tuple of (pixel_values (num_frames, 3, H, W), metadata dict).
    """
    try:
        import decord
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        fps = vr.get_avg_fps()
        indices = sample_frames(total, num_frames, strategy, fps=fps)
        frames = vr.get_batch(indices).asnumpy()
    except ImportError:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            indices = sample_frames(total, num_frames, strategy, fps=fps)

            frames_list = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame)
            cap.release()

            if not frames_list:
                raise RuntimeError(f"Failed to read frames from {video_path}")

            import numpy as np
            frames = np.stack(frames_list)
        except ImportError:
            raise ImportError(
                "Video loading requires 'decord' or 'opencv-python'. "
                "Install with: pip install decord  or  pip install opencv-python"
            )

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    pixel_values = torch.stack([transform(f) for f in frames])
    metadata = {
        "total_frames": total,
        "sampled_indices": indices,
        "fps": fps,
        "num_sampled": len(indices),
    }
    return pixel_values, metadata


@VISION_ENCODERS.register("video")
class VideoEncoder(nn.Module):
    """Video encoder that processes video frames through an image encoder
    and applies temporal modeling.

    Pipeline: Frame sampling -> Per-frame encoding -> Temporal position encoding
              -> Temporal pooling -> Video tokens for LLM input.
    """

    def __init__(
        self,
        config: VisionConfig,
        num_frames: int = 8,
        temporal_pool_type: str = "attention",
    ):
        super().__init__()
        self.config = config
        self.num_frames = num_frames

        self.frame_encoder = build_vision_encoder(config)
        hidden_size = config.hidden_size

        self.token_generator = VideoTokenGenerator(hidden_size, max_frames=64)
        self.temporal_pool = TemporalPooling(
            hidden_size, num_heads=8, pool_type=temporal_pool_type,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video frames into pooled visual tokens.

        Args:
            pixel_values: (B, num_frames, 3, H, W) or (num_frames, 3, H, W) video frames.

        Returns:
            (B, num_tokens, hidden_size) pooled video features.
        """
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        B, T, C, H, W = pixel_values.shape
        flat_frames = pixel_values.reshape(B * T, C, H, W)

        with torch.set_grad_enabled(self.training):
            frame_features = self.frame_encoder(flat_frames)

        N, D = frame_features.shape[1], frame_features.shape[2]
        frame_features = frame_features.reshape(B, T, N, D)

        frame_features = self.token_generator(frame_features, T)
        pooled = self.temporal_pool(frame_features)

        return pooled

    def encode_video(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        strategy: str = "uniform",
    ) -> torch.Tensor:
        """Convenience method: load a video file and encode it.

        Args:
            video_path: Path to video file.
            num_frames: Override for number of frames to sample.
            strategy: Frame sampling strategy.

        Returns:
            (1, num_tokens, hidden) video features.
        """
        n = num_frames or self.num_frames
        pixel_values, _ = load_video_frames(
            video_path, num_frames=n, strategy=strategy,
            image_size=self.config.image_size,
        )
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=next(self.parameters()).dtype)
        return self(pixel_values)
