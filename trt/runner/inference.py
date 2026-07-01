from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from trt.config.execution_mode import ExecutionMode
from trt.config.stage_config import StageConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve


@dataclass
class InferenceStageResult:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _stage_timing_key(stage_cfg: StageConfig) -> str:
    if stage_cfg.engine_subdir == "visual":
        return "vision"
    return stage_cfg.engine_subdir or f"stage_{stage_cfg.stage_id}"


def _run_hook_for_mode(mode: ExecutionMode) -> str:
    return {
        ExecutionMode.EAGER: "run_eager",
        ExecutionMode.SERIALIZED: "run_serialized",
        ExecutionMode.IN_MEMORY: "run_trt",
    }[mode]


class InferenceStageRunner:
    def __init__(self, stage_cfg: StageConfig):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    @torch.no_grad()
    def run(self, ctx: EdgeContext) -> InferenceStageResult:
        upstream = [ctx.stage_results[i] for i in self.stage_cfg.input_sources]
        if self.hooks.process_inputs:
            resolve(self.hooks.process_inputs)(ctx, upstream, {})

        hook_name = _run_hook_for_mode(ctx.execution_mode)
        hook_path = getattr(self.hooks, hook_name)
        if not hook_path:
            raise ValueError(
                f"inference stage {self.stage_cfg.stage_id} ({self.stage_cfg.kind}) "
                f"missing {hook_name} hook for {ctx.execution_mode.value}"
            )

        t0 = time.perf_counter()
        result = resolve(hook_path)(ctx)
        ctx.inference.stage_ms[_stage_timing_key(self.stage_cfg)] = (time.perf_counter() - t0) * 1000

        if self.hooks.after_stage:
            resolve(self.hooks.after_stage)(ctx, result)

        ctx.stage_results[self.stage_cfg.stage_id] = result
        return result
