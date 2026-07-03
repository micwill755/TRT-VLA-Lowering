from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.data import pack_state
from trt.executor.models.groot.helpers import make_embodiment_id


def preprocess(ctx: EdgeContext) -> None:
    """Normalize model inputs before the inference stage loop (GR00T).

    Same tensors as export preprocess; see ``groot/export/preprocess.py`` for
    libero shape examples. Writes into ``ctx.inference`` instead of export_state.
    """
    tokenized_data = ctx.model_inputs["tokenized_data"]
    infer = ctx.inference
    infer.tokenized = {
        "input_ids": tokenized_data["input_ids"],          # [1, T]
        "attention_mask": tokenized_data["attention_mask"],  # [1, T]
    }
    infer.pixel_values = tokenized_data["pixel_values"].to(
        device=ctx.device,
        dtype=torch.float16,
    )  # [2, 3, 224, 224] fp16
    state = pack_state(
        ctx.model_inputs["state"],  # [7] libero proprio
        max_state_dim=ctx.policy.config.max_state_dim,  # 64
        device=ctx.device,
    )  # [1, 1, 64]
    infer.action_side = {
        "state": state.to(device=ctx.device, dtype=torch.float16).contiguous(),
        "embodiment_id": make_embodiment_id(ctx.policy, state, ctx.device).contiguous(),  # [1]
    }
