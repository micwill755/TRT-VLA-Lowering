from __future__ import annotations

import torch
import torch_tensorrt

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule
from trt.compile import compile_trt_module
from trt.plugin.plugin_utils import patch_vision_attention, restore_attention

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    """Prepare the pixel tensor and eager vision wrapper for the vision stage.

    ``GridVisionExportModule`` accepts either NCHW or HWC pixels and normalizes
    internally before calling the HF vision tower. Keeping NCHW here matches the
    TRT engine input contract and the shape used in ``test_vla.py``.
    """

    # Eagle model owns the SigLIP vision tower and projector used by GR00T.
    eagle = ctx.model.backbone.eagle_model
    vision = eagle.vision_model

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
    visual_module = GridVisionExportModule(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=pixel_values,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=ctx.dtype)
    
    # patch vision attention
    hidden_states = vision.vision_model.embeddings(pixel_values)
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    patched = patch_vision_attention(
        vision.vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    try:
        trt_engine = compile_trt_module(
            visual_module,
            (pixel_values,),
            {**ctx.trt_settings, "use_python_runtime": True},
        )
    finally:
        restore_attention(patched)
    return {
        "pixel_values": pixel_values,
        "visual_module": visual_module,
        "trt_engine": trt_engine,
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
    visual = inputs["visual_module"]
    pixel_values = inputs["pixel_values"]
    image_embs = visual(pixel_values)

    return {
        "tensors": {
            "image_embs": image_embs,
        },
        "metadata": {
            "backend": "eager",
        },
    }

def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    trt_engine = inputs["trt_engine"]
    pixel_values = inputs["pixel_values"].contiguous()

    with torch.no_grad():
        image_embs = trt_engine(pixel_values)

    return {
        "tensors": {
            "image_embs": image_embs,
        },
        "metadata": {
            "backend": "in_memory_trt",
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

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    ctx.inference.image_embs = result["tensors"]["image_embs"]
    return result