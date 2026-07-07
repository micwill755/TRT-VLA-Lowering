from __future__ import annotations

import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from trt.context import EdgeContext


def preprocess(ctx: EdgeContext) -> dict:
    """PI05 export batch prep (same tensors as inference pipeline preprocess)."""
    model_inputs = ctx.model_inputs
    images, img_masks = ctx.policy._preprocess_images(model_inputs)

    tokens = model_inputs[OBS_LANGUAGE_TOKENS].to(device=ctx.device, dtype=torch.long)
    masks = model_inputs[OBS_LANGUAGE_ATTENTION_MASK].to(device=ctx.device, dtype=torch.bool)

    pixel_values = torch.cat(
        [img.to(device=ctx.device, dtype=ctx.dtype) for img in images],
        dim=0,
    ).contiguous()

    tokenizer = getattr(ctx.profile, "text_tok", None) or getattr(ctx.profile, "text_tokenizer", None)
    if tokenizer is not None:
        ctx.export_state["tokenizer"] = tokenizer

    return {
        "images": images,
        "img_masks": img_masks,
        "tokens": tokens,
        "masks": masks,
        "pixel_values": pixel_values,
    }


def postprocess(ctx: EdgeContext, stage_outputs: dict) -> None:
    pass
