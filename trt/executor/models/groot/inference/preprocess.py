from __future__ import annotations

import torch

from trt.data import pack_state
from trt.executor.models.groot.helpers import make_embodiment_id
from trt.inference.context import InferenceContext


def preprocess(ctx: InferenceContext) -> None:
    tokenized_data = ctx.model_inputs["tokenized_data"]
    ctx.tokenized = {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
    }
    ctx.pixel_values = tokenized_data["pixel_values"].to(
        device=ctx.device,
        dtype=torch.float16,
    )
    state = pack_state(
        ctx.model_inputs["state"],
        max_state_dim=ctx.policy.config.max_state_dim,
        device=ctx.device,
    )
    ctx.action_side = {
        "state": state.to(device=ctx.device, dtype=torch.float16).contiguous(),
        "embodiment_id": make_embodiment_id(ctx.policy, state, ctx.device).contiguous(),
    }
