from __future__ import annotations

import torch

from trt.compile import save_trt_engine_module
from trt.context import EdgeContext
from trt.modules.export.vision import GridVisionExportModule
from trt.plugin.plugin_utils import infer_smolvlm_seq_len, patch_vision_attention, restore_attention
from trt.vision import nchw_to_hwc


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    vlm = ctx.model.vlm_with_expert.get_vlm_model()
    vision = vlm.vision_model
    connector = vlm.connector

    pixel_values = inputs["pixel_values"].to(
        device=ctx.device,
        dtype=ctx.dtype,
    ).contiguous()

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


def export(ctx: EdgeContext, inputs: dict) -> dict:
    pixel_values = inputs["pixel_values"]
    visual_module = inputs["visual_module"]

    vlm = ctx.model.vlm_with_expert.get_vlm_model()
    vision = vlm.vision_model
    language = vlm.text_model

    batch_size, vision_seq_len = infer_smolvlm_seq_len(vision, pixel_values)
    seq_len = int(visual_module.output_seq_len)
    vocab_size = int(language.config.vocab_size)
    image_token_id = int(
        ctx.model.vlm_with_expert.processor.tokenizer.convert_tokens_to_ids("<image>")
    )
    images_hwc = nchw_to_hwc(pixel_values)

    patched = patch_vision_attention(
        vision,
        batch_size=batch_size,
        seq_len=vision_seq_len,
        name="SigLIP",
        allow_attention_mask=True,
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
