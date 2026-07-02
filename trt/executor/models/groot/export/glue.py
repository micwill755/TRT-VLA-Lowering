from __future__ import annotations

import torch


def vision_to_language(ctx, upstream, stage_inputs):
    vision = upstream[0]
    export_state = getattr(ctx, "export_state", {})
    tokenized = export_state["tokenized"]
    image_token_id = int(vision.metadata["image_token_id"])
    hidden_size = int(vision.metadata["output_hidden_size"])
    num_rows = int((tokenized["input_ids"] == image_token_id).sum().item())

    return {
        **stage_inputs,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "image_embs": torch.zeros(
            num_rows,
            hidden_size,
            device=ctx.device,
            dtype=torch.float16,
        ),
        "image_token_id": image_token_id,
        "seq_len_per_image": int(vision.metadata["config_seq_len"]),
    }


def language_to_action_context(ctx, upstream, stage_inputs):
    language = upstream[0]
    meta = language.metadata
    return {
        **stage_inputs,
        "lm_hidden_states": torch.zeros(
            int(meta["batch_size"]),
            int(meta["max_seq_len"]),
            int(meta["hidden_size"]),
            device=ctx.device,
            dtype=torch.float16,
        ),
        "language_inputs": meta.get("language_inputs"),
    }


def action_context_to_action(ctx, upstream, stage_inputs):
    context = upstream[0]
    action_side = ctx.export_state["action_side"]
    meta = context.metadata
    return {
        **stage_inputs,
        "context_embs": torch.zeros(
            int(meta["batch_size"]),
            int(meta["context_seq_len"]),
            int(meta["context_hidden_size"]),
            device=ctx.device,
            dtype=torch.float16,
        ),
        "state": action_side["state"],
        "embodiment_id": action_side["embodiment_id"],
        "language_inputs": meta.get("language_inputs"),
    }
