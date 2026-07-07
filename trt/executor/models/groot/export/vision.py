from __future__ import annotations

import torch
import torch_tensorrt

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule
from trt.compile import compile_trt_module
from trt.plugin.plugin_utils import patch_vision_attention, restore_attention
from trt.serialize import SerializedTRTEngine
from trt.compile import save_trt_engine_module
from trt.vision import (
    nchw_to_hwc,
)

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    eagle = ctx.model.backbone.eagle_model
    vision = eagle.vision_model

    pixel_values = inputs["pixel_values"]
    pixel_values = pixel_values.to(
        device=ctx.device,
        dtype=ctx.dtype,
    ).contiguous()
    
    visual_module = GridVisionExportModule(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=pixel_values,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
        vision_kwargs={},
    ).eval().to(device=ctx.device, dtype=ctx.dtype)
    
    return {
        "pixel_values": pixel_values,
        "visual_module": visual_module
    }

def export(ctx: EdgeContext, inputs: dict) -> dict:
    pixel_values = inputs["pixel_values"]
    visual_module = inputs["visual_module"]

    eagle = ctx.model.backbone.eagle_model
    vision = eagle.vision_model

    # patch vision attention
    hidden_states = vision.vision_model.embeddings(pixel_values)
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    image_token_id = int(getattr(eagle, "image_token_index", eagle.config.image_token_index))
    vocab_size = int(eagle.language_model.config.vocab_size)
    images_hwc = nchw_to_hwc(pixel_values_nchw)  # [B, H, W, 3]

    patched = patch_vision_attention(
        vision.vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    
    try:
        engine_path = save_trt_engine_module(
            visual_module,
            (images_hwc,),
            ctx.engine_root / "visual",
            engine_file="visual.engine",
            model_type="visual",
            component="vision",
            input_names=["pixel_values"],
            output_names=["visual_embeds"],
            example_output=None,
            extra_config={
                vocab_size=vocab_size,
                image_token_id=image_token_id,
                seq_len=seq_len,
            },
            trt_settings=ctx.trt_settings,
        )
    finally:
        restore_attention(patched)

    return {
        "engine_path": engine_path
    }

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    # TODO:
    return result