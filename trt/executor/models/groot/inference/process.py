from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.data import pack_state
from trt.executor.models.groot.helpers import make_embodiment_id


def preprocess(ctx: EdgeContext) -> dict:
    device = ctx.device
    dtype = ctx.dtype
    tokenized_data = ctx.model_inputs["tokenized_data"]

    pixel_values = tokenized_data["pixel_values"].to(
        device=device, dtype=dtype
    ).contiguous()
    state = pack_state(
        ctx.model_inputs["state"],
        max_state_dim=ctx.policy.config.max_state_dim,
        device=device,
    ).to(device=device, dtype=dtype).contiguous()
    embodiment_id = make_embodiment_id(ctx.policy, state, device, torch.long)

    return {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "pixel_values": pixel_values,
        "state": state,
        "embodiment_id": embodiment_id,
    }


def postprocess(ctx: EdgeContext, stage_outputs: dict) -> None:
    pass
