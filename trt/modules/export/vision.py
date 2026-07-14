# trt/modules/export/vision.py

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch
import torch.nn as nn

from trt.vision import hwc_to_nchw, is_nchw_pixel_values

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ExportModule(nn.Module):
    """Base TRT trace target: probe static shapes once, flatten output to [N, H]."""

    cast_output_to_input_dtype: bool
    output_num_tokens: int
    output_hidden_size: int

    def _finalize_output(self, features: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if self.cast_output_to_input_dtype and features.dtype != out_dtype:
            features = features.to(out_dtype)
        if features.ndim == 3:
            # [B, S, H] -> [B*S, H]
            return features.reshape(-1, features.shape[-1])
        if features.ndim == 2:
            # already [N, H] (token-pooling encoders)
            return features
        raise ValueError(f"Expected 2D or 3D features, got shape {tuple(features.shape)}")

# ---------------------------------------------------------------------------
# Grid vision (PI0.5 / GR00T / SmolVLA VitRunner path)
# ---------------------------------------------------------------------------

class GridVisionExportModule(ExportModule):
    """pixels [B,H,W,C] -> fixed patch grid -> [B*seq_len, lm_hidden]"""

    def __init__(
        self,
        *,
        vision_model: nn.Module,
        projector: nn.Module,
        sample_pixel_values: torch.Tensor,
        select_layer: int = -1,
        pixel_shuffle: bool = False,
        downsample_ratio: float = 0.5,
        force_float32_input: bool = False,
        cast_output_to_input_dtype: bool = False,
        vision_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.projector = projector
        self.select_layer = int(select_layer)
        self.pixel_shuffle = bool(pixel_shuffle)
        self.downsample_ratio = float(downsample_ratio)
        self.force_float32_input = bool(force_float32_input)
        self.cast_output_to_input_dtype = bool(cast_output_to_input_dtype)
        self.vision_kwargs = dict(vision_kwargs or {})

        with torch.no_grad():
            sample = sample_pixel_values
            if self.force_float32_input and sample.dtype != torch.float32:
                sample = sample.to(torch.float32)

            vit_embeds = self._select_vision_features(self._run_vision(sample))
            self.seq_len = int(vit_embeds.shape[1])
            self.hidden_size = int(vit_embeds.shape[2])

            if self.pixel_shuffle:
                self._init_pixel_shuffle_shape()
                vit_embeds = self._apply_pixel_shuffle(vit_embeds)

            projected = self.projector(self._projector_input(vit_embeds))
            self.batch_size = int(projected.shape[0])
            self.output_seq_len = int(projected.shape[1])
            self.output_hidden_size = int(projected.shape[2])
            self.output_num_tokens = self.batch_size * self.output_seq_len

    def _run_vision(self, images: torch.Tensor):
        pixel_values = hwc_to_nchw(images) if not is_nchw_pixel_values(images) else images
        kwargs = dict(self.vision_kwargs)
        kwargs["pixel_values"] = pixel_values
        kwargs["output_hidden_states"] = self.select_layer != -1
        kwargs.setdefault("return_dict", True)
        return self.vision_model(**kwargs)

    def _select_vision_features(self, out):
        if self.select_layer == -1:
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return out.hidden_states[self.select_layer]

    def _projector_input(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        proj_dtype = next(self.projector.parameters()).dtype
        if vit_embeds.dtype != proj_dtype:
            return vit_embeds.to(proj_dtype)
        return vit_embeds

    def _init_pixel_shuffle_shape(self):
        side = int(self.seq_len ** 0.5)
        if side * side != self.seq_len:
            raise ValueError(f"Expected square vision sequence, got seq_len={self.seq_len}")
        self.grid_w = side
        self.grid_h = side
        self.out_w = int(self.grid_w * self.downsample_ratio)
        self.out_h = int(self.grid_h * self.downsample_ratio)
        self.hidden_after_first_view = int(self.hidden_size / self.downsample_ratio)
        self.shuffle_hidden = int(self.hidden_size / (self.downsample_ratio * self.downsample_ratio))

    def _apply_pixel_shuffle(self, x):
        n = x.shape[0]
        x = x.reshape(n, self.grid_w, self.out_h, self.hidden_after_first_view)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, self.out_h, self.out_w, self.shuffle_hidden)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(n, -1, self.shuffle_hidden)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out_dtype = pixel_values.dtype
        images = pixel_values
        if self.force_float32_input and images.dtype != torch.float32:
            images = images.to(torch.float32)

        vit_embeds = self._select_vision_features(self._run_vision(images))
        if self.pixel_shuffle:
            vit_embeds = self._apply_pixel_shuffle(vit_embeds)

        features = self.projector(self._projector_input(vit_embeds))
        return self._finalize_output(features, out_dtype)

# ---------------------------------------------------------------------------
# Token pooling vision (MolmoAct2 and similar)
# ---------------------------------------------------------------------------

class TokenPoolingExportModule(ExportModule):
    """media + pooling_indices -> sparse prompt-token embeddings [num_valid, H]

    Generic wrapper for encoders like MolmoAct2 ``vision_backbone`` that:
      - take batched crop/patch tensors + pooling index table
      - return one row per valid prompt image token (invalid slots filtered inside encoder)
    """

    def __init__(
        self,
        *,
        encoder: nn.Module | TokenPoolingEncoder,
        sample_media: torch.Tensor,
        sample_pooling_indices: torch.Tensor,
        cast_output_to_input_dtype: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.cast_output_to_input_dtype = bool(cast_output_to_input_dtype)

        with torch.no_grad():
            out = self._encode(sample_media, sample_pooling_indices)
            if out.ndim != 2:
                raise ValueError(
                    f"TokenPoolingExportModule expects encoder output [N, H], got {tuple(out.shape)}"
                )

            # static input dims for TRT
            self.batch_size = int(sample_media.shape[0])
            self.media_shape = tuple(sample_media.shape[1:])
            self.max_pooled_tokens = int(sample_pooling_indices.shape[1])
            self.pool_dim = int(sample_pooling_indices.shape[2])

            self.output_num_tokens = int(out.shape[0])
            self.output_hidden_size = int(out.shape[-1])

    def _encode(self, media: torch.Tensor, pooling_indices: torch.Tensor) -> torch.Tensor:
        return self.encoder(media, pooling_indices)

    def forward(self, media: torch.Tensor, pooling_indices: torch.Tensor) -> torch.Tensor:
        out_dtype = media.dtype
        features = self._encode(media, pooling_indices)
        return self._finalize_output(features, out_dtype)