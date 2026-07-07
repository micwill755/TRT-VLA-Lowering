from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.plan import ExportPlan
from trt.io_spec import GROOT_EDGE_IO
from trt.language import make_action_context_module
from trt.context import EdgeContext

DEFAULT_ACTION_CONTEXT_TRT_SETTINGS: dict[str, Any] = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "truncate_double": True,
    "immutable_weights": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
    "use_fp32_acc": True,
}

def preprocess(ctx: EdgeContext, stage_inputs: dict) -> ExportPlan:
    dtype = torch.float16
    lm_hidden_states = stage_inputs["lm_hidden_states"].to(
        device=ctx.device,
        dtype=dtype,
    ).contiguous()
    batch_size = int(lm_hidden_states.shape[0])
    max_seq_len = int(lm_hidden_states.shape[1])
    hidden_size = int(lm_hidden_states.shape[2])

    vlln = ctx.model.action_head.vlln
    if getattr(vlln, "weight", None) is not None:
        context_hidden_size = int(vlln.weight.shape[0])
    else:
        context_hidden_size = int(
            ctx.model.backbone.eagle_model.language_model.config.hidden_size
        )

    module = make_action_context_module(ctx.model, device=ctx.device, dtype=dtype)
    io = GROOT_EDGE_IO.action_context

    return ExportPlan(
        module=module,
        sample_inputs=(lm_hidden_states,),
        input_names=tuple(io.input_names),
        output_names=tuple(io.output_names),
        engine_dir=ctx.engine_root / "action_context",
        engine_file="context.engine",
        model_type="action_context",
        component="context",
        trt_settings=DEFAULT_ACTION_CONTEXT_TRT_SETTINGS,
        cleanup_modules=(module,),
        args={
            "batch_size": batch_size,
            "extra_config": {
                "engine_role": "action_context",
                "max_seq_len": max_seq_len,
                "hidden_size": hidden_size,
                "context_hidden_size": context_hidden_size,
            },
            "language_inputs": stage_inputs.get("language_inputs"),
            "tensor_aliases": {"context_embs": "vl_embs"},
        },
    )


def compile(plan: ExportPlan) -> Path:
    return save_trt_engine_module(
        plan.module,
        plan.sample_inputs,
        plan.engine_dir,
        engine_file=plan.engine_file,
        model_type=plan.model_type or "action_context",
        component=plan.component or "context",
        input_names=list(plan.input_names),
        output_names=list(plan.output_names),
        example_output=None,
        extra_config=plan.args["extra_config"],
        trt_settings=plan.trt_settings,
    )


def postprocess(ctx: EdgeContext, plan: ExportPlan) -> dict:
    del ctx
    extra = plan.args["extra_config"]
    return {
        "batch_size": plan.args["batch_size"],
        "language_inputs": plan.args.get("language_inputs"),
        "context_seq_len": int(extra["max_seq_len"]),
        "context_hidden_size": int(extra["context_hidden_size"]),
    }
