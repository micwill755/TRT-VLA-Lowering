from __future__ import annotations

import torch

from trt.compile import compile_trt_module
from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.executor.models.smolvla.load.serialize import SerializedSmolVLAVision
from trt.modules.export.vision import GridVisionExportModule
from trt.plugin.plugin_utils import infer_smolvlm_seq_len, patch_vision_attention, restore_attention
from trt.serialize import SerializedTRTEngine


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    vlm = ctx.model.vlm_with_expert.get_vlm_model()
    vision = vlm.vision_model
    connector = vlm.connector
    pixel_values = inputs["pixel_values"]

    visual_module = GridVisionExportModule(
        vision_model=vision,
        projector=connector,
        sample_pixel_values=pixel_values,
        select_layer=-1,
        pixel_shuffle=False,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=ctx.dtype)

    return {
        "pixel_values": pixel_values,
        "visual_module": visual_module,
    }


def compile(ctx: EdgeContext, inputs: dict) -> dict:
    pixel_values = inputs["pixel_values"]
    visual_module = inputs["visual_module"]

    vision = ctx.model.vlm_with_expert.get_vlm_model().vision_model
    batch_size, seq_len = infer_smolvlm_seq_len(vision, pixel_values)

    patched = patch_vision_attention(
        vision,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
        allow_attention_mask=True,
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
    serialized_vision = SerializedSmolVLAVision(
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
