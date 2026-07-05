from __future__ import annotations

import time

import torch

from trt.config.stage_config import PipelineConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.hooks = config.hooks

    @torch.no_grad()
    def run(self, ctx: EdgeContext, inputs: dict) -> EdgeContext:
        torch.manual_seed(ctx.seed)
        if ctx.device == "cuda":
            torch.cuda.manual_seed_all(ctx.seed)
        
        t0 = time.perf_counter()

        if self.hooks.preprocess:
            resolve(self.hooks.preprocess)(ctx)

        result = {}
        stg_input, stg_output = inputs, {}
        for stage_cfg in self.config.stages:
            print("Executing {}".format(stage_cfg.name))
            stg_input = stg_output
            runner = resolve(stage_cfg.runner)(stage_cfg)
            stg_output = runner.run(ctx, stg_input)

        if self.hooks.postprocess:
            result = resolve(self.hooks.postprocess)(ctx, result)

        execution_time = time.perf_counter() - t0
        print("Pipeline complete in {}", execution_time)

        # TODO: what do we need to return after pipeline ? 
        return result