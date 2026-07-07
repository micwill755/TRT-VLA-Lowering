from __future__ import annotations

import torch

from trt.data import pack_state
from trt.executor.models.groot.helpers import make_embodiment_id
from trt.context import EdgeContext

def preprocess(ctx: EdgeContext) -> None:
    """Normalize model inputs before the stage loop (GR00T)."""
    # [1, T] int64 chat tokens (T ≈ prompt len)
    tokenized_data = ctx.model_inputs["tokenized_data"]
    # [1, T] int64 1 = attend, 0 = pad
    ctx.export_state["tokenized"] = {
        "input_ids": tokenized_data["input_ids"],          # [1, T]
        "attention_mask": tokenized_data["attention_mask"],  # [1, T]
    }
    # [2, 3, 224, 224]  float32  two cameras, NCHW
    ctx.export_state["pixel_values"] = tokenized_data["pixel_values"].to(
        device=ctx.device,
        dtype=torch.float16,
    )
    # [7] float32 libero proprio (D = 7)
    state = pack_state(
        ctx.model_inputs["state"],  # [7] libero proprio
        max_state_dim=ctx.policy.config.max_state_dim,  # 64
        device=ctx.device,
    ) 
    ctx.export_state["action_side"] = {
        "state": state,
        "embodiment_id": make_embodiment_id(ctx.policy, state, ctx.device, ctx.dtype),  # [1], e.g. [31]
    }
    ctx.export_state["tokenizer"] = ctx.profile.text_tok

def postprocess(ctx: EdgeContext) -> None:
    pass