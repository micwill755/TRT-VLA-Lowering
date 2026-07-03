from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule
from trt.runner.inference import InferenceStageResult
from trt.vision import nchw_to_hwc

def run(ctx: EdgeContext) -> InferenceStageResult:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx)

def _run_eager(ctx: EdgeContext) -> InferenceStageResult:
    infer = ctx.inference
    pixel_values = infer.pixel_values.to(device=ctx.device, dtype=torch.float16).contiguous()
    images_hwc = nchw_to_hwc(pixel_values)
    eagle = ctx.model.backbone.eagle_model
    visual = GridVisionExportModule(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=images_hwc,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=torch.float16)
    image_embs = visual(images_hwc)
    infer.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})

def _run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    module = ctx.handles.serialized.vision
    if module is None:
        raise RuntimeError("serialized TRT backend missing vision module")
    image_embs = module(ctx.inference.pixel_values.contiguous())
    ctx.inference.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})

def _run_trt(ctx: EdgeContext) -> InferenceStageResult:
    module = ctx.handles.in_memory.vision
    if module is None:
        raise RuntimeError("in-memory TRT backend missing vision module")
    image_embs = module(ctx.inference.pixel_values.contiguous())
    ctx.inference.image_embs = image_embs
    return InferenceStageResult(tensors={"image_embs": image_embs})