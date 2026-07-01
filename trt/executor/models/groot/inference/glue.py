from __future__ import annotations

from trt.inference.context import InferenceContext, LanguageOutputs


def vision_to_language(ctx: InferenceContext, upstream, stage_inputs: dict) -> dict:
    """Wire vision stage outputs into language prefill inputs."""
    vision = upstream[0]
    ctx.image_embs = vision.tensors["image_embs"]
    return {
        **stage_inputs,
        "input_ids": ctx.tokenized["input_ids"],
        "attention_mask": ctx.tokenized["attention_mask"],
        "image_embs": ctx.image_embs,
    }


def language_to_action_context(ctx: InferenceContext, upstream, stage_inputs: dict) -> dict:
    """Wire language stage outputs into action-context / rollout."""
    language = upstream[0]
    lm_hidden = language.tensors["lm_hidden_states"]
    ctx.lm = LanguageOutputs(lm_hidden_states=lm_hidden)
    if "language_inputs" in language.metadata:
        ctx.language_inputs = language.metadata["language_inputs"]
    return {
        **stage_inputs,
        "lm_hidden_states": lm_hidden,
        "language_inputs": ctx.language_inputs,
        "state": ctx.action_side["state"],
        "embodiment_id": ctx.action_side["embodiment_id"],
    }
