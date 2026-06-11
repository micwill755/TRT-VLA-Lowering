import copy
import pathlib
import json

from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt

import logging

logger = logging.getLogger(__name__)

class GROOTVisualEmbed(nn.Module):
    """
    Semantic GR00T visual wrapper.

    This calls Eagle's high-level extract_feature helper, which already returns
    the projected visual-token embeddings consumed by the prompt/language path.
    It is useful for eager comparisons, but extract_feature hides the selected
    hidden-state path plus the optional pixel-shuffle projection path. In that
    path Eagle derives reshape sizes from vit_embeds.shape using Python math:
    sqrt, int(...), a float downsample_ratio, and -1 reshape inference. Those
    ops are simple semantically, but they can be awkward for torch.export and
    TensorRT when they are produced from symbolic tensor shapes inside the
    helper.
    """
    def __init__(self, groot):
        super().__init__()
        self.eagle_model = groot.backbone.eagle_model

    def forward(self, pixel_values):
        return self.eagle_model.extract_feature(pixel_values)

class PI05VisualEmbed(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.paligemma_with_expert = core.paligemma_with_expert

    def forward(self, image):
        return self.paligemma_with_expert.embed_image(image)

class SmolVLAVisualEmbed(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert

    def forward(self, image):
        return self.vlm_with_expert.embed_image(image)

class PixelOnlyWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, args):
        return self.wrapped(**args)

class VisualFixedInput(nn.Module):
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
                # TODO, look into method for removing forced fp32 conversion.
                sample = sample.to(torch.float32)

            vit_embeds = self._select_vision_features(self._run_vision(sample))

            self.seq_len = int(vit_embeds.shape[1])
            self.hidden_size = int(vit_embeds.shape[2])

            if self.pixel_shuffle:
                self._init_pixel_shuffle_shape()
                vit_embeds = self._apply_pixel_shuffle(vit_embeds)

            projected = self.projector(vit_embeds)
            self.output_seq_len = int(projected.shape[1])
            self.output_hidden_size = int(projected.shape[2])

    def _run_vision(self, pixel_values: torch.Tensor):
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
        n = x.shape[0]

        x = x.reshape(n, self.grid_w, self.out_h, self.hidden_after_first_view)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.reshape(n, self.out_h, self.out_w, self.shuffle_hidden)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.reshape(n, -1, self.shuffle_hidden)

    def forward(self, pixel_values):
        out_dtype = pixel_values.dtype
        
        if self.force_float32_input and pixel_values.dtype != torch.float32:
            # TODO, look into method for removing forced fp32 conversion.
            pixel_values = pixel_values.to(torch.float32)

        vit_embeds = self._select_vision_features(self._run_vision(pixel_values))

        if self.pixel_shuffle:
            vit_embeds = self._apply_pixel_shuffle(vit_embeds)

        features = self.projector(vit_embeds)

        if self.cast_output_to_input_dtype and features.dtype != out_dtype:
            features = features.to(out_dtype)

        return features