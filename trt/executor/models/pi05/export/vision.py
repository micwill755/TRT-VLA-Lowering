from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule
from trt.plugin.plugin_utils import patch_vision_attention, restore_attention
from trt.compile import save_trt_engine_module
from trt.vision import nchw_to_hwc


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    paligemma = ctx.model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    projector = paligemma.multi_modal_projector

    pixel_values = inputs["pixel_values"].to(
        device=ctx.device,
        dtype=ctx.dtype,
    ).contiguous()

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


def export(ctx: EdgeContext, inputs: dict) -> dict:
    pixel_values = inputs["pixel_values"]
    visual_module = inputs["visual_module"]

    paligemma = ctx.model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    language = paligemma.language_model

    hidden_states = vision.embeddings(pixel_values.float())
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    vocab_size = int(language.config.vocab_size)
    image_token_id = int(getattr(paligemma.config, "image_token_index", 257152))
    images_hwc = nchw_to_hwc(pixel_values)

    patched = patch_vision_attention(
        vision,
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
            model_type="vit",
            component="vision",
            input_names=["pixel_values"],
            output_names=["visual_embeds"],
            extra_config={
                "vocab_size": vocab_size,
                "image_token_id": image_token_id,
                "builder_config": {
                    "seq_len": seq_len,
                },
            },
            trt_settings=ctx.trt_settings,
        )
    finally:
        restore_attention(patched)

    with torch.no_grad():
        image_embs = visual_module(pixel_values)

    return {
        "engine_path": engine_path,
        "tensors": {
            "image_embs": image_embs,
        },
        "metadata": {
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "image_token_id": image_token_id,
        },
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    return result
