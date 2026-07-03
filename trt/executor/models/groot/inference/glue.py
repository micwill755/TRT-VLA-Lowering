from __future__ import annotations

from trt.context import EdgeContext


def vision_to_language(ctx: EdgeContext, upstream, stage_inputs: dict) -> dict:
    vision = upstream[0]
    ctx.inference.image_embs = vision.tensors["image_embs"]
    return {
        **stage_inputs,
        "input_ids": ctx.inference.tokenized["input_ids"],
        "attention_mask": ctx.inference.tokenized["attention_mask"],
        "image_embs": ctx.inference.image_embs,
    }


def language_to_action_context(ctx: EdgeContext, upstream, stage_inputs: dict) -> dict:
    language = upstream[0]
    lm_hidden = language.tensors["lm_hidden_states"]
    ctx.inference.lm_hidden_states = lm_hidden
    if "logits" in language.tensors:
        ctx.inference.logits = language.tensors["logits"]
    if "language_inputs" in language.metadata:
        ctx.inference.language_inputs = language.metadata["language_inputs"]
    return {
        **stage_inputs,
        "lm_hidden_states": lm_hidden,
        "language_inputs": ctx.inference.language_inputs,
        "state": ctx.inference.action_side["state"],
        "embodiment_id": ctx.inference.action_side["embodiment_id"],
    }
