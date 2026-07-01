from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from trt.config.stage_config import StageConfig
from trt.hooks.resolve import resolve
from trt.inference.backends import InferenceBackend
from trt.inference.context import InferenceContext


@dataclass
class InferenceStageResult:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _stage_timing_key(stage_cfg: StageConfig) -> str:
    if stage_cfg.engine_subdir == "visual":
        return "vision"
    return stage_cfg.engine_subdir or f"stage_{stage_cfg.stage_id}"


class InferenceStageRunner:
    def __init__(self, stage_cfg: StageConfig):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    @torch.no_grad()
    def run(
        self,
        ctx: InferenceContext,
        backend: InferenceBackend,
    ) -> InferenceStageResult:
        upstream = [ctx.stage_results[i] for i in self.stage_cfg.input_sources]
        stage_inputs: dict[str, Any] = {}
        if self.hooks.process_inputs:
            stage_inputs = resolve(self.hooks.process_inputs)(ctx, upstream, stage_inputs)

        if not self.hooks.run:
            raise ValueError(
                f"inference stage {self.stage_cfg.stage_id} ({self.stage_cfg.kind}) missing run hook"
            )

        t0 = time.perf_counter()
        result = resolve(self.hooks.run)(ctx, backend, stage_inputs)
        ctx.stage_ms[_stage_timing_key(self.stage_cfg)] = (time.perf_counter() - t0) * 1000

        if self.hooks.after_stage:
            resolve(self.hooks.after_stage)(ctx, result)

        ctx.stage_results[self.stage_cfg.stage_id] = result
        return result
