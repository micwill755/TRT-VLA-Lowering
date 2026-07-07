from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from trt.config.stage_config import StageConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve

class BenchmarkRunner:
    def __init__(self, stage_cfg: StageConfig):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks
    
    @torch.no_grad()
    def run(self, ctx: EdgeContext, inputs: dict) -> Any:
        # preprocess any inputs before run
        if self.hooks.get("preprocess"):
            resolve(self.hooks["preprocess"])(ctx, inputs)
        # start timer
        t0 = time.perf_counter()
        result = resolve(self.hooks["execute"])(ctx)
        # record total time of execution
        execution_time = time.perf_counter() - t0

        if self.hooks.get("postprocess"):
            resolve(self.hooks["postprocess"])(ctx, result)

        return {
            "execution_time": execution_time,
            "result": result
        }