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
    def __init__(self, groot):
        super().__init__()
        self.eagle_model = groot.backbone.eagle_model

    def forward(self, pixel_values):
        return self.eagle_model.extract_feature(pixel_values)

class PixelOnlyWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, args):
        return self.wrapped(**args)

class GROOTVisualFixedInput(nn.Module):
    def __init__(self, groot, sample_pixel_values: torch.Tensor):
        super().__init__()

        self.eagle_model = groot.backbone.eagle_model
        self.vision_model = self.eagle_model.vision_model
        self.mlp1 = self.eagle_model.mlp1

        self.select_layer = int(getattr(self.eagle_model, "select_layer", -1))
        self.use_pixel_shuffle = bool(getattr(self.eagle_model, "use_pixel_shuffle", False))
        self.downsample_ratio = float(getattr(self.eagle_model, "downsample_ratio", 0.5))

        with torch.no_grad():
            out = self.vision_model(
                pixel_values=sample_pixel_values,
                output_hidden_states=self.select_layer != -1,
                return_dict=True,
            )

            if self.select_layer == -1:
                vit_embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            else:
                vit_embeds = out.hidden_states[self.select_layer]

        self.seq_len = int(vit_embeds.shape[1])
        self.hidden_size = int(vit_embeds.shape[2])

        if self.use_pixel_shuffle:
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

    def _pixel_shuffle_fixed(self, x):
        n = x.shape[0]

        x = x.reshape(
            n,
            self.grid_w,
            self.out_h,
            self.hidden_after_first_view,
        )
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.reshape(
            n,
            self.out_h,
            self.out_w,
            self.shuffle_hidden,
        )
        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def forward(self, pixel_values):
        if self.select_layer == -1:
            out = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
            vit_embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        else:
            out = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            vit_embeds = out.hidden_states[self.select_layer]

        if self.use_pixel_shuffle:
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0],
                self.grid_w,
                self.grid_h,
                self.hidden_size,
            )
            vit_embeds = self._pixel_shuffle_fixed(vit_embeds)
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0],
                -1,
                vit_embeds.shape[-1],
            )

        return self.mlp1(vit_embeds)

class PI05VisualEmbed(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.paligemma_with_expert = core.paligemma_with_expert

    def forward(self, image):
        return self.paligemma_with_expert.embed_image(image)

class FP16CastWrapper(nn.Module):
    def __init__(self, trt_model):
        super().__init__()
        self.trt_model = trt_model

    def forward(self, image):
        return self.trt_model(image.to(torch.float16))

class SmolVLAVisualEmbed(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert

    def forward(self, image):
        return self.vlm_with_expert.embed_image(image)