from __future__ import annotations

import torch

from trt.export.groot import make_visual_fixed_input
from trt.inference.backends import EagerBackend, InferenceBackend
from trt.inference.context import InferenceContext
from trt.runner.inference import InferenceStageResult
from trt.vision import nchw_to_hwc


def run(
    ctx: InferenceContext,
    backend: InferenceBackend,
    stage_inputs: dict,
) -> InferenceStageResult:
    del stage_inputs
    if isinstance(backend, EagerBackend):
        images_hwc = nchw_to_hwc(
            ctx.pixel_values.to(device=ctx.device, dtype=torch.float16).contiguous()
        )
        visual = make_visual_fixed_input(
            ctx.model,
            images_hwc,
            device=ctx.device,
            dtype=torch.float16,
        )
        image_embs = visual(images_hwc)
    elif ctx.vision_module is not None:
        image_embs = ctx.vision_module(ctx.pixel_values.contiguous())
    else:
        image_embs = backend.run_vision(ctx, ctx.pixel_values)

    ctx.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})
