from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from . import root


def discover(policy: nn.Module) -> nn.Module:
    found = root(policy)
    if found is None:
        raise RuntimeError("PI05 vision adapter matched a policy without paligemma_with_expert")
    return found.paligemma_with_expert.paligemma.model.vision_tower.eval()


def wrap_vision(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    from trt.modules.export.vision import GridVisionExportModule

    found = root(policy)
    if found is None:
        raise RuntimeError("PI05 vision patch matched a policy without paligemma_with_expert")
    paligemma = found.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower.float()
    pixel_values = inputs["pixel_values"]
    sample = pixel_values.float() if pixel_values.dtype != torch.float32 else pixel_values
    return GridVisionExportModule(
        vision_model=vision,
        projector=paligemma.multi_modal_projector,
        sample_pixel_values=sample,
        select_layer=-1,
        pixel_shuffle=False,
        downsample_ratio=0.5,
        force_float32_input=True,
    ).eval()
