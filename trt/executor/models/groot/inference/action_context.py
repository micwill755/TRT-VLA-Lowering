from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.language import make_action_context_module
from trt.runner.inference import InferenceStageResult

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    context_module = ContextProjectionExportModule(
        ctx.model.backbone.eagle_linear,
        ctx.model.action_head.vlln,
        ctx.model.action_head.vl_self_attention,
    ).eval()

    return {
        "context_module": context_module
    }

def run(ctx: EdgeContext) -> InferenceStageResult:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx)

def _run_eager(ctx: EdgeContext) -> InferenceStageResult:
    lm_hidden = inputs["lm_hidden"]
    context_module = inputs["context_module"]
    # pre action transformation - action context
    context_embs = ctx.model.backbone.eagle_linear(lm_hidden)
    context_embs = context_module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    
    return {
        "context_embs": context_embs
    }


def _run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    lm_hidden = _lm_hidden(ctx)
    module = ctx.handles.serialized.action_context
    if module is None:
        context_embs = lm_hidden
    else:
        context_embs = module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.inference.context_embs})


def _run_trt(ctx: EdgeContext) -> InferenceStageResult:
    lm_hidden = _lm_hidden(ctx)
    module = ctx.handles.in_memory.action_context
    if module is None:
        context_embs = lm_hidden
    else:
        context_embs = module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.inference.context_embs})
