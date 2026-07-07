from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from trt.config.stage_config import StageConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve

class InferenceRunner:
    def __init__(self, stage_cfg: StageConfig):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    @torch.no_grad()
    def run(self, ctx: EdgeContext, inputs: dict) -> dict:
        t0 = time.perf_counter()

        # if the stage has a preprocess function, make sure we update inputs
        prepared = inputs 
        if self.hooks.get("preprocess"):
            prepared = resolve(self.hooks["preprocess"])(ctx)

        result = resolve(self.hooks["execute"])(ctx, prepared)

        if self.hooks.get("postprocess"):
            result = resolve(self.hooks["postprocess"])(ctx, result)

        execution_time = time.perf_counter() - t0
        print("Runner: execution complete in {}", execution_time)

        # TODO: what do we need to return after pipeline ? 
        return result
