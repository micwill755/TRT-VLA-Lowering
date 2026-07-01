from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.action_plan import ActionExportPlan


def compile_action_plan(plan: ActionExportPlan) -> Path:
    sample_inputs = tuple(
        x.contiguous() if isinstance(x, torch.Tensor) else x
        for x in plan.sample_inputs
    )
    with torch.no_grad():
        example_output = plan.module(*sample_inputs)

    return save_trt_engine_module(
        plan.module,
        sample_inputs,
        plan.engine_dir,
        engine_file=plan.engine_file,
        model_type=plan.model_type or "action",
        component=plan.component or "action",
        input_names=list(plan.input_names),
        output_names=list(plan.output_names),
        example_output=example_output,
        extra_config=plan.extra_config,
        trt_settings=plan.trt_settings,
    )
