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

class GROOTVisualFixedInput(nn.Module):
    """
    Export-stable GR00T visual wrapper.

    This spells out the same visual embedding path as extract_feature using
    fixed metadata inferred from a representative image: selected vision layer,
    optional pixel shuffle dimensions, and the final mlp1 projection. The output
    is still GR00T visual embeddings; the difference is that the shape-sensitive
    work is explicit and static for TensorRT export.

    The problematic part is not the SigLIP vision tower itself. It is the
    post-vision pixel-shuffle path in extract_feature:

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(batch, h, w, -1)
        pixel_shuffle uses int(h * scale), int(c / scale), and int(c / scale**2)
        vit_embeds = vit_embeds.reshape(batch, -1, hidden)
        vit_embeds = mlp1(vit_embeds)

    This wrapper precomputes grid/output/hidden sizes from sample_pixel_values so
    the exported graph sees fixed reshape -> permute -> reshape -> mlp1 ops,
    instead of Python int/sqrt/float scale math derived from symbolic shapes.
    """
    def __init__(self, groot, sample_pixel_values: torch.Tensor):
        super().__init__()

        self.eagle_model = groot.backbone.eagle_model
        self.vision_model = self.eagle_model.vision_model
        self.mlp1 = self.eagle_model.mlp1

        # select layer is the output (hidden states) we want to pass to the mlp
        # e.g. if select_layer = -4, then the fourth-from-last vision layer output is what gets passed to mlp1.
        '''
            pixel_values

            -> vision_model
            -> choose hidden state via select_layer
            -> optional pixel_shuffle (turns spatial resolution down and channels up
            e.g downsample_ratio = 0.5 on [2, 16, 16, hidden] -> [2, 8, 8, hidden * 4] -> [2, 64, hidden * 4]
            -> mlp1
            -> visual embeddings for GROOT

        '''
        self.select_layer = int(getattr(self.eagle_model, "select_layer", -1))
        self.use_pixel_shuffle = bool(getattr(self.eagle_model, "use_pixel_shuffle", False))
        self.downsample_ratio = float(getattr(self.eagle_model, "downsample_ratio", 0.5))

        with torch.no_grad():
            # Run one sample through the vision model to infer the static vision feature
            # shape used by this wrapper: sequence length and hidden size before mlp1.
            # If select_layer != -1, we need all hidden states so we can pick that layer;
            # otherwise last_hidden_state is enough.

            '''
            output_hidden_states=self.select_layer != -1
            means:

            select_layer == -1   -> output_hidden_states=False
                                    returns only last_hidden_state

            select_layer != -1   -> output_hidden_states=True
                                    returns all hidden_states
                                    then we choose hidden_states[select_layer]

            '''
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