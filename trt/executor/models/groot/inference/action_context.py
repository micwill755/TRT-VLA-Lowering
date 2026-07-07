from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.compile import compile_trt_module
from trt.executor.models.groot.load.serialize import SerializedGrootActionContext
from trt.modules.export.language import ContextProjectionExportModule
from trt.pipelines.parity import maybe_override_upstream
from trt.serialize import SerializedTRTEngine

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    inputs = maybe_override_upstream(ctx, "action_context", inputs)

    device, dtype = ctx.device, ctx.dtype

    # upstream language post: [B, S, 2048]
    lm_hidden = inputs["tensors"]["lm_hidden"]
    lm_hidden = lm_hidden.to(device=device, dtype=dtype).contiguous()

    # eagle_linear -> vlln -> vl_self_attention (test_vla.py action context block)
    context_module = ContextProjectionExportModule(
        ctx.model.backbone.eagle_linear,
        ctx.model.action_head.vlln,
        ctx.model.action_head.vl_self_attention,
    ).eval().to(device=device, dtype=dtype)

    return {
        "lm_hidden": lm_hidden,
        "context_module": context_module,
    }


def compile(ctx: EdgeContext, inputs: dict) -> dict:
    context_module = inputs["context_module"]
    lm_hidden = inputs["lm_hidden"]

    trt_engine = compile_trt_module(
        context_module,
        (lm_hidden,),
        {**ctx.trt_settings, "use_python_runtime": True},
    )

    return {
        "trt_engine": trt_engine,
    }


def load(ctx: EdgeContext, inputs: dict) -> dict:
    serialized_action_context = SerializedGrootActionContext(
        SerializedTRTEngine(ctx.engine_root / "action_context")
    )
    return {
        "serialized_engine": serialized_action_context,
    }


def execute(ctx: EdgeContext, inputs: dict) -> dict:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx, inputs)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx, inputs)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx, inputs)

    raise ValueError(f"unsupported execution mode: {ctx.execution_mode}")


def _run_eager(ctx: EdgeContext, inputs: dict) -> dict:
    context_module = inputs["context_module"]
    lm_hidden = inputs["lm_hidden"]

    with torch.no_grad():
        context_embs = context_module(lm_hidden)

    return {
        "tensors": {
            "context_embs": context_embs,
        },
        "metadata": {
            "backend": "eager",
        },
    }


def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    trt_engine = inputs["trt_engine"]
    lm_hidden = inputs["lm_hidden"]

    with torch.no_grad():
        context_embs = trt_engine(lm_hidden)

    return {
        "tensors": {
            "context_embs": context_embs,
        },
        "metadata": {
            "backend": "in_memory_trt",
        },
    }


def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    module = inputs["serialized_engine"]

    lm_hidden = inputs["lm_hidden"].contiguous()

    with torch.no_grad():
        context_embs = module(lm_hidden)

    return {
        "tensors": {
            "context_embs": context_embs,
        },
        "metadata": {
            "backend": "serialized_trt",
        },
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    ctx.inference.context_embs = result["tensors"]["context_embs"]
    return result
