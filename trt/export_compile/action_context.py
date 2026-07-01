from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.action_context_plan import ActionContextExportPlan


def compile_action_context_plan(plan: ActionContextExportPlan) -> Path:
    with torch.no_grad():
        example_output = plan.module(*plan.sample_inputs)

    return save_trt_engine_module(
        plan.module,
        plan.sample_inputs,
        plan.engine_dir,
        engine_file=plan.engine_file,
        model_type=plan.model_type or "action_context",
        component=plan.component or "action_context",
        input_names=list(plan.input_names),
        output_names=list(plan.output_names),
        example_output=example_output,
        extra_config=plan.extra_config,
        trt_settings=plan.trt_settings,
    )
