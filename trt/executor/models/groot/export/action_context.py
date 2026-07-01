from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.plan import ExportPlan
from trt.io_spec import GROOT_EDGE_IO
from trt.language import make_action_context_module
from trt.runner.base import StageContext

DEFAULT_ACTION_CONTEXT_TRT_SETTINGS: dict[str, Any] = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "truncate_double": True,
    "immutable_weights": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
    "use_fp32_acc": True,
}


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
    dtype = torch.float16
    lm_hidden_states = stage_inputs["lm_hidden_states"].to(
        device=ctx.device,
        dtype=dtype,
    ).contiguous()

    module = GROOTContextProjectionWrapper(
        copy.deepcopy(core.backbone.eagle_linear).to(device=device, dtype=dtype).eval(),
        copy.deepcopy(core.action_head.vlln).to(device=device, dtype=dtype).eval(),
        copy.deepcopy(core.action_head.vl_self_attention).to(device=device, dtype=dtype).eval(),
    ).eval()
    
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
            "extra_config": {
                "engine_role": "action_context",
                "max_seq_len": int(lm_hidden_states.shape[1]),
                "hidden_size": int(lm_hidden_states.shape[2]),
            },
            "language_inputs": stage_inputs.get("language_inputs"),
            "tensor_aliases": {"context_embs": "vl_embs"},
        },
    )


def compile(plan: ExportPlan, eager_output) -> Path:
    output = eager_output[0] if isinstance(eager_output, (tuple, list)) else eager_output
    extra_config = dict(plan.args["extra_config"])
    extra_config["context_hidden_size"] = int(output.shape[-1])

    return save_trt_engine_module(
        plan.module,
        plan.sample_inputs,
        plan.engine_dir,
        engine_file=plan.engine_file,
        model_type=plan.model_type or "action_context",
        component=plan.component or "context",
        input_names=list(plan.input_names),
        output_names=list(plan.output_names),
        example_output=eager_output,
        extra_config=extra_config,
        trt_settings=plan.trt_settings,
    )


def metadata(ctx: StageContext, plan: ExportPlan, output) -> dict:
    del ctx
    vl_embs = output[0] if isinstance(output, (tuple, list)) else output
    return {
        "language_inputs": plan.args.get("language_inputs"),
        "context_seq_len": int(vl_embs.shape[1]),
        "context_hidden_size": int(vl_embs.shape[2]),
    }
