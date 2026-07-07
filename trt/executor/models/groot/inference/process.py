from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.data import pack_state
from trt.executor.models.groot.helpers import make_embodiment_id

def preprocess(ctx: EdgeContext) -> dict:
    tokenized_data = ctx.model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"].to(device=ctx.device, dtype=torch.long)
    attention_mask = tokenized_data["attention_mask"].to(device=ctx.device, dtype=torch.long)
    pixel_values = tokenized_data["pixel_values"].to(device=ctx.device, dtype=ctx.dtype)
    state = pack_state(
        ctx.model_inputs["state"],  # [7] libero proprio
        max_state_dim=64,  # 64
        device=ctx.device,
    ) 
    state = state.to(device=ctx.device, dtype=ctx.dtype).contiguous()
    embodiment_id = make_embodiment_id(ctx.policy, state, ctx.device, torch.long)

    ctx.inference.action_side["state"] = state.detach()
    ctx.inference.action_side["embodiment_id"] = embodiment_id.detach()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "state": state,
        "embodiment_id": embodiment_id,
    }

def postprocess(ctx: EdgeContext, stage_outputs: dict) -> None:
    pass