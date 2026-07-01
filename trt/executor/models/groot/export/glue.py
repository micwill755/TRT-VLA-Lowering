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


def language_to_action(ctx, upstream, stage_inputs):
    language, _vision = upstream[0], upstream[1]
    export_state = getattr(ctx, "export_state", {})
    action_side = export_state["action_side"]

    return {
        **stage_inputs,
        "lm_hidden_states": language.tensors["hidden_states"],
        "language_inputs": language.metadata["language_inputs"],
        "state": action_side["state"],
        "embodiment_id": action_side["embodiment_id"],
    }
