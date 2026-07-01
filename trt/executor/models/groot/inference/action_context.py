from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.language import make_action_context_module
from trt.runner.inference import InferenceStageResult


def _lm_hidden(ctx: EdgeContext) -> torch.Tensor:
    if ctx.inference.lm is not None and ctx.inference.lm.lm_hidden_states is not None:
        return ctx.inference.lm.lm_hidden_states
    raise RuntimeError("action_context stage requires language lm_hidden_states")


def run_eager(ctx: EdgeContext) -> InferenceStageResult:
    lm_hidden = _lm_hidden(ctx)
    context_module = make_action_context_module(
        ctx.model,
        device=ctx.device,
        dtype=torch.float16,
    )
    context_embs = context_module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.inference.context_embs})


def run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    lm_hidden = _lm_hidden(ctx)
    module = ctx.handles.serialized.action_context
    if module is None:
        context_embs = lm_hidden
    else:
        context_embs = module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.inference.context_embs})


def run_trt(ctx: EdgeContext) -> InferenceStageResult:
    lm_hidden = _lm_hidden(ctx)
    module = ctx.handles.in_memory.action_context
    if module is None:
        context_embs = lm_hidden
    else:
        context_embs = module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.inference.context_embs})
