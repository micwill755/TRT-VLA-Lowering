from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.config.execution_mode import ExecutionMode
from trt.modules.export.vision import GridVisionExportModule
from trt.compile import compile_trt_module
from trt.plugin.plugin_utils import patch_vision_attention, restore_attention
from trt.executor.models.pi05.load.serialize import SerializedPi05Vision
from trt.serialize import SerializedTRTEngine


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    """Prepare PaliGemma SigLIP vision + projector for the PI05 vision stage."""
    paligemma = ctx.model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    projector = paligemma.multi_modal_projector
    pixel_values = inputs["pixel_values"]

    visual_module = GridVisionExportModule(
        vision_model=vision,
        projector=projector,
        sample_pixel_values=pixel_values,
        select_layer=-1,
        pixel_shuffle=False,
        downsample_ratio=0.5,
        force_float32_input=True,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=ctx.dtype)

    return {
        "pixel_values": pixel_values,
        "visual_module": visual_module,
    }


def compile(ctx: EdgeContext, inputs: dict) -> dict:
    pixel_values = inputs["pixel_values"]
    visual_module = inputs["visual_module"]

    vision = ctx.model.paligemma_with_expert.paligemma.model.vision_tower
    hidden_states = vision.embeddings(pixel_values.float())
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

    patched = patch_vision_attention(
        vision,
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
        "trt_engine": trt_engine,
    }


def load(ctx: EdgeContext, inputs: dict) -> dict:
    serialized_vision = SerializedPi05Vision(
        SerializedTRTEngine(ctx.engine_root / "visual")
    )
    return {
        "serialized_engine": serialized_vision,
    }


def execute(ctx: EdgeContext, inputs: dict) -> dict:
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

    with torch.no_grad():
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
    module = inputs["serialized_engine"]
    pixel_values = inputs["pixel_values"].contiguous()

    with torch.no_grad():
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
