from __future__ import annotations

import torch

from trt.data import pack_state
from trt.export.groot import make_embodiment_id
from trt.runner.base import StageContext

def preprocess(ctx: StageContext) -> None:
    """Normalize model inputs before the stage loop (GR00T)."""
    tokenized_data = ctx.model_inputs["tokenized_data"]
    ctx.handles["tokenized"] = {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
    }
    ctx.handles["pixel_values"] = tokenized_data["pixel_values"].to(
        device=ctx.device,
        dtype=torch.float16,
    )
    state = pack_state(
        ctx.model_inputs["state"],
        max_state_dim=ctx.policy.config.max_state_dim,
        device=ctx.device,
    )
    ctx.handles["action_side"] = {
        "state": state,
        "embodiment_id": make_embodiment_id(ctx.policy, state, ctx.device),
    }
