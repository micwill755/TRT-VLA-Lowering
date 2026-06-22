import copy
import pathlib
import json

from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt

import logging

logger = logging.getLogger(__name__)

VIT_ENGINE_INPUT_NAME = "input"
VIT_ENGINE_OUTPUT_NAME = "output"

def nchw_to_hwc(pixel_values: torch.Tensor) -> torch.Tensor:
    """Convert processor-style NCHW pixel values to HWC for VisualFixedInput / VitRunner."""
    if pixel_values.ndim != 4:
        raise ValueError(f"Expected 4D pixel_values, got shape {tuple(pixel_values.shape)}")
    return pixel_values.permute(0, 2, 3, 1).contiguous()

def hwc_to_nchw(images: torch.Tensor) -> torch.Tensor:
    """Convert HWC ``[batch, H, W, C]`` to NCHW for HuggingFace SigLIP vision models."""
    if images.ndim != 4:
        raise ValueError(f"Expected 4D images, got shape {tuple(images.shape)}")
    return images.permute(0, 3, 1, 2).contiguous()

def is_nchw_pixel_values(pixel_values: torch.Tensor) -> bool:
    """True when channels are in dim 1 (processor-style NCHW)."""
    return (
        pixel_values.ndim == 4
        and pixel_values.shape[1] in (1, 3, 4)
        and pixel_values.shape[-1] not in (1, 3, 4)
    )

def vit_visual_edge_config(
    *,
    vocab_size: int,
    image_token_id: int,
    seq_len: int,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> dict[str, Any]:
    """Build visual/config.json fields consumed by VitRunner."""
    builder_config: dict[str, Any] = {"seq_len": int(seq_len)}
    if image_mean is not None:
        builder_config["image_mean"] = list(image_mean)
    if image_std is not None:
        builder_config["image_std"] = list(image_std)
    return {
        "vocab_size": int(vocab_size),
        "image_token_id": int(image_token_id),
        "builder_config": builder_config,
    }

class PixelOnlyWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, args):
        return self.wrapped(**args)

class VisualFixedInput(nn.Module):
    """Fixed-shape vision tower + projector for TRT export.

    Expects **preprocessed** HWC input ``[batch, H, W, C]`` (matches C++ ``normalizeImage`` output),
    not raw uint8 images. Output is flattened for Edge-LLM multimodal runners that index
    vision rows sequentially (Phi-style ``embeddingLookupWithImageInsertion``).

    Typical GROOT flow (2 cameras stacked on the vision batch dim)::

        input  [batch, H, W, C]   e.g. [2, 224, 224, 3]  (HWC, preprocessed FP16/FP32)
            -> vision_model
        vit_embeds    [batch, seq_len, vit_hidden]
            -> optional pixel_shuffle
            -> projector
        features      [batch, output_seq_len, lm_hidden]
            -> reshape
        return        [batch * output_seq_len, lm_hidden]
    """

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
            sample = sample_pixel_values  # [batch, H, W, C]
            if self.force_float32_input and sample.dtype != torch.float32:
                # TODO, look into method for removing forced fp32 conversion.
                sample = sample.to(torch.float32)

            vit_embeds = self._select_vision_features(self._run_vision(sample))
            # vit_embeds: [batch, seq_len, vit_hidden]

            self.seq_len = int(vit_embeds.shape[1])
            self.hidden_size = int(vit_embeds.shape[2])

            if self.pixel_shuffle:
                self._init_pixel_shuffle_shape()
                vit_embeds = self._apply_pixel_shuffle(vit_embeds)
                # vit_embeds: [batch, output_seq_len, shuffle_hidden]

            projected = self.projector(vit_embeds)
            # projected: [batch, output_seq_len, lm_hidden]
            self.batch_size = int(projected.shape[0])
            self.output_seq_len = int(projected.shape[1])
            self.output_hidden_size = int(projected.shape[2])
            self.output_num_tokens = self.batch_size * self.output_seq_len

    def _run_vision(self, images: torch.Tensor):
        # TRT / VitRunner boundary is HWC; HuggingFace SigLIP expects NCHW.
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

    def _init_pixel_shuffle_shape(self):
        side = int(self.seq_len ** 0.5)
        if side * side != self.seq_len:
            raise ValueError(f"Expected square vision sequence, got seq_len={self.seq_len}")

        self.grid_w = side
        self.grid_h = side
        self.out_w = int(self.grid_w * self.downsample_ratio)
        self.out_h = int(self.grid_h * self.downsample_ratio)
        self.hidden_after_first_view = int(self.hidden_size / self.downsample_ratio)
        self.shuffle_hidden = int(
            self.hidden_size / (self.downsample_ratio * self.downsample_ratio)
        )

    def _apply_pixel_shuffle(self, x):
        # x: [batch, seq_len, vit_hidden]  where seq_len = grid_h * grid_w
        n = x.shape[0]

        x = x.reshape(n, self.grid_w, self.out_h, self.hidden_after_first_view)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.reshape(n, self.out_h, self.out_w, self.shuffle_hidden)
        x = x.permute(0, 2, 1, 3).contiguous()

        # [batch, output_seq_len, shuffle_hidden]
        return x.reshape(n, -1, self.shuffle_hidden)

    def forward(self, input):
        # input: [batch, H, W, C]  preprocessed HWC, fixed shape for TRT (VitRunner binding: "input")
        images = input
        out_dtype = images.dtype
        
        if self.force_float32_input and images.dtype != torch.float32:
            # TODO, look into method for removing forced fp32 conversion.
            images = images.to(torch.float32)

        vit_embeds = self._select_vision_features(self._run_vision(images))
        # vit_embeds: [batch, seq_len, vit_hidden]

        if self.pixel_shuffle:
            vit_embeds = self._apply_pixel_shuffle(vit_embeds)
            # vit_embeds: [batch, output_seq_len, shuffle_hidden]

        features = self.projector(vit_embeds)
        # features: [batch, output_seq_len, lm_hidden]

        if self.cast_output_to_input_dtype and features.dtype != out_dtype:
            features = features.to(out_dtype)

        # Flatten batch + tokens for Edge-LLM: [batch * output_seq_len, lm_hidden]
        return features.reshape(-1, features.shape[-1])