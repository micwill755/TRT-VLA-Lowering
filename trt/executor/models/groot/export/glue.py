from __future__ import annotations


def vision_to_language(ctx, upstream, stage_inputs):
    vision = upstream[0]
    export_state = getattr(ctx, "export_state", {})
    tokenized = export_state["tokenized"]

    return {
        **stage_inputs,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "image_embs": vision.tensors["image_embs"],
        "image_token_id": vision.metadata["image_token_id"],
        "seq_len_per_image": vision.metadata["config_seq_len"],
    }


def language_to_action_context(ctx, upstream, stage_inputs):
    del ctx
    language = upstream[0]
    lm_hidden = language.tensors.get("lm_hidden_states", language.tensors["hidden_states"])
    return {
        **stage_inputs,
        "lm_hidden_states": lm_hidden,
        "language_inputs": language.metadata.get("language_inputs"),
    }


def action_context_to_action(ctx, upstream, stage_inputs):
    context = upstream[0]
    action_side = ctx.export_state["action_side"]
    context_embs = context.tensors.get("context_embs", context.tensors["vl_embs"])
    return {
        **stage_inputs,
        "context_embs": context_embs,
        "state": action_side["state"],
        "embodiment_id": action_side["embodiment_id"],
        "language_inputs": context.metadata.get("language_inputs"),
    }
