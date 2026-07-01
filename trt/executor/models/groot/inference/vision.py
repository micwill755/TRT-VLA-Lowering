from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule
from trt.runner.inference import InferenceStageResult
from trt.vision import nchw_to_hwc


def run_eager(ctx: EdgeContext) -> InferenceStageResult:
    infer = ctx.inference
    images_hwc = nchw_to_hwc(
        infer.pixel_values.to(device=ctx.device, dtype=torch.float16).contiguous()
    )
    eagle = ctx.model.backbone.eagle_model
    visual = GridVisionExportModule(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=images_hwc,
        select_layer=int(eagle.select_layer),
        pixel_shuffle=bool(eagle.use_pixel_shuffle),
        downsample_ratio=float(eagle.downsample_ratio),
        force_float32_input=False,
        cast_output_to_input_dtype=False,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=torch.float16)
    image_embs = visual(images_hwc)
    infer.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})


def run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    module = ctx.handles.serialized.vision
    if module is None:
        raise RuntimeError("serialized TRT backend missing vision module")
    image_embs = module(ctx.inference.pixel_values.contiguous())
    ctx.inference.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})


def run_trt(ctx: EdgeContext) -> InferenceStageResult:
    module = ctx.handles.in_memory.vision
    if module is None:
        raise RuntimeError("in-memory TRT backend missing vision module")
    image_embs = module(ctx.inference.pixel_values.contiguous())
    ctx.inference.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})
