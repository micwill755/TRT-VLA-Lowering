from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    """Prepare the pixel tensor and eager vision wrapper for the vision stage.

    ``GridVisionExportModule`` accepts either NCHW or HWC pixels and normalizes
    internally before calling the HF vision tower. Keeping NCHW here matches the
    TRT engine input contract and the shape used in ``test_vla.py``.
    """

    # Eagle model owns the SigLIP vision tower and projector used by GR00T.
    eagle = ctx.model.backbone.eagle_model

    # Prefer the explicit stage input, but fall back to ctx.inference scratch state.
    # Shape: [B, C, H, W], e.g. [1, 3, 224, 224].
    pixel_values = inputs.get("pixel_values", ctx.inference.pixel_values)
    if pixel_values is None:
        raise ValueError("vision stage missing pixel_values")

    # Vision/TRT path uses fp16 pixels on the target device.
    # Contiguous layout keeps both HF eager and TRT runtime input ABI stable.
    pixel_values = pixel_values.to(
        device=ctx.device,
        dtype=torch.float16,
    ).contiguous()

    # The eager backend needs a Python wrapper around:
    #   vision_model -> selected hidden state -> optional pixel shuffle -> projector.
    #
    # Input shape to wrapper forward:
    #   [B, C, H, W] NCHW pixels
    #
    # Output shape from wrapper:
    #   [B * S_visual, H_lm]
    # Example from your run:
    #   [512, 2048]
    visual = GridVisionExportModule(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=pixel_values,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=ctx.dtype)

    # Return only stage-local prepared objects. ``visual`` is used by eager only;
    # TRT/serialized backends ignore it and call their loaded engine handles.
    return {
        "pixel_values": pixel_values,
        "visual": visual,
    }

def execute(ctx: EdgeContext, inputs: dict) -> dict:
    # TODO: do we need sperate functions for each execution mode 
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx, inputs)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx, inputs)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx, inputs)

    raise ValueError(f"unsupported execution mode: {ctx.execution_mode}")

def _run_eager(ctx: EdgeContext, inputs: dict) -> dict:
    # Eager uses the Python wrapper from preprocess.
    visual = inputs["visual"]

    # Pixel tensor shape: [B, C, H, W].
    pixel_values = inputs["pixel_values"]

    # Run SigLIP + projector in PyTorch.
    # Output shape: [B * S_visual, H_lm], e.g. [512, 2048].
    image_embs = visual(pixel_values)

    return {
        "tensors": {
            "image_embs": image_embs,
        },
        "metadata": {
            "backend": "eager",
        },
    }

def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    # Serialized backend is the loaded ``visual.engine`` runner.
    module = ctx.handles.serialized.vision
    if module is None:
        raise RuntimeError("serialized TRT backend missing vision module")

    # Engine input shape matches exported vision ABI: [B, C, H, W] fp16.
    pixel_values = inputs["pixel_values"].contiguous()

    # Engine output shape matches eager wrapper: [B * S_visual, H_lm].
    image_embs = module(pixel_values)

    return {
        "tensors": {
            "image_embs": image_embs,
        },
        "metadata": {
            "backend": "serialized_trt",
        },
    }

def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    # In-memory backend is the Torch-TensorRT module created during same-process compile.
    module = ctx.handles.in_memory.vision
    if module is None:
        raise RuntimeError("in-memory TRT backend missing vision module")

    # Engine input shape matches exported vision ABI: [B, C, H, W] fp16.
    pixel_values = inputs["pixel_values"].contiguous()

    # Engine output shape matches eager wrapper: [B * S_visual, H_lm].
    image_embs = module(pixel_values)

    return {
        "tensors": {
            "image_embs": image_embs,
        },
        "metadata": {
            "backend": "in_memory_trt",
        },
    }

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    tensors = result["tensors"]

    ctx.inference.pixel_values = tensors.get("pixel_values")
    ctx.inference.image_embs = tensors.get("image_embs")

    return result