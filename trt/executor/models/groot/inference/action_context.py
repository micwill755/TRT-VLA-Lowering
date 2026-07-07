from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.language import make_action_context_module
from trt.runner.inference import InferenceStageResult
from trt.compile import compile_trt_module
from trt.modules.export.language import CausalLMExportModule, ContextProjectionExportModule

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    lm_hidden = inputs["tensors"]["lm_hidden"]  # from language stage
    lm_hidden = lm_hidden.to(device=ctx.device, dtype=ctx.dtype).contiguous()

    # input hidden_states  [B, S, 2048]   ← language context_hidden (lm_hidden_eager or trt_out[1])
    context_module = ContextProjectionExportModule(
        model.backbone.eagle_linear, # [B, S, 1536],
        model.action_head.vlln, # [B, S, 1536] (LayerNorm),
        model.action_head.vl_self_attention # [B, S, 1536] (vl_self_attention (SelfAttentionTransformer)) -> [output],
    ).eval().to(device=ctx.device, dtype=ctx.dtype)

    trt_engine = compile_trt_module(
        context_module, (lm_hidden,), {**ctx.trt_settings, "use_python_runtime": True}
    )

    return {
        "lm_hidden": lm_hidden,
        "context_module": context_module,
        "trt_engine": trt_engine,
    }

def execute(ctx: EdgeContext, inputs: dict) -> dict:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx, inputs)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx, inputs)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx, inputs)

def _run_eager(ctx: EdgeContext, inputs: dict) -> dict:
    lm_hidden = inputs["lm_hidden"]
    context_module = inputs["context_module"]
    context_embs = context_module(lm_hidden)
    
    return {
        "eager_context_embs": context_embs
    }

def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    lm_hidden = inputs["lm_hidden"]
    trt_engine = inputs["action_context_engine"]
    context_embs = trt_engine(lm_hidden)
    
    return {
        "trt_context_embs": context_embs
    }

def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    lm_hidden = _lm_hidden(ctx)
    module = ctx.handles.serialized.action_context
    if module is None:
        context_embs = lm_hidden
    else:
        context_embs = module(lm_hidden)
    ctx.inference.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.inference.context_embs})