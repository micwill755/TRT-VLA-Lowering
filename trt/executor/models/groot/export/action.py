from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.diffusion_builders import build_groot_diffusion_export_params
from trt.hooks.export.plan import ExportPlan
from trt.modules.export.diffusion import DEFAULT_DIFFUSION_TRT_SETTINGS
from trt.runner.base import StageContext


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
    context_embs = stage_inputs["context_embs"].to(
        device=ctx.device,
        dtype=torch.float16,
    ).contiguous()
    state = stage_inputs["state"]
    embodiment_id = stage_inputs["embodiment_id"]

    spec = build_groot_diffusion_export_params(
        ctx.model,
        context_embs=context_embs,
        state=state,
        embodiment_id=embodiment_id,
        device=ctx.device,
        trt_settings=DEFAULT_DIFFUSION_TRT_SETTINGS,
    )

    return ExportPlan(
        module=spec.diffusion_module,
        sample_inputs=spec.sample_inputs,
        input_names=tuple(spec.io.input_names),
        output_names=tuple(spec.io.output_names),
        engine_dir=ctx.engine_root / "action",
        engine_file=spec.engine_file,
        model_type=spec.model_type,
        component=spec.component,
        trt_settings=spec.trt_settings,
        cleanup_modules=(spec.diffusion_module,),
        args={
            "extra_config": spec.extra_config,
            "context_seq_len": int(context_embs.shape[1]),
            "context_hidden_size": int(context_embs.shape[2]),
            "language_inputs": stage_inputs.get("language_inputs"),
        },
    )


def compile(plan: ExportPlan, eager_output) -> Path:
    return save_trt_engine_module(
        plan.module,
        plan.sample_inputs,
        plan.engine_dir,
        engine_file=plan.engine_file,
        model_type=plan.model_type or "action",
        component=plan.component or "diffusion",
        input_names=list(plan.input_names),
        output_names=list(plan.output_names),
        example_output=eager_output,
        extra_config=plan.args["extra_config"],
        trt_settings=plan.trt_settings,
    )


def metadata(ctx: StageContext, plan: ExportPlan, output) -> dict:
    del ctx, output
    args = plan.args
    return {
        "language_inputs": args.get("language_inputs"),
        "context_seq_len": args["context_seq_len"],
        "context_hidden_size": args["context_hidden_size"],
    }
