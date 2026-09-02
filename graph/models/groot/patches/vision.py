from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from . import root


def discover(policy: nn.Module) -> nn.Module:
    found = root(policy)
    if found is None:
        raise RuntimeError("GR00T vision adapter matched a policy without backbone.eagle_model")
    return found.backbone.eagle_model.vision_model.eval()


def wrap_vision(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    from trt.modules.export.vision import GridVisionExportModule

    found = root(policy)
    if found is None:
        raise RuntimeError("GR00T vision patch matched a policy without backbone.eagle_model")
    eagle = found.backbone.eagle_model
    return GridVisionExportModule(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=inputs["pixel_values"],
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=getattr(eagle, "downsample_ratio", 0.5),
        vision_kwargs={},
    ).eval()
