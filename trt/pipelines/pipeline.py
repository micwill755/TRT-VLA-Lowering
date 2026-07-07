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

        if self.hooks.get("preprocess"):
            resolve(self.hooks["preprocess"])(ctx)

        result = {}
        stage_outputs = {}
        for stage_cfg in self.config.stages:
            print("Executing {}".format(stage_cfg.name))
            # using this upstream approach allows for optional DAG and streamline 
            inputs = [stage_outputs[i] for i in stage_cfg.input_sources]
            runner = resolve(stage_cfg.runner)(stage_cfg)
            stg_output = runner.run(ctx, inputs)
            stage_outputs[stage_cfg.stage_id] = stg_output

        if self.hooks.get("postprocess"):
            result = resolve(self.hooks["postprocess"])(ctx, result)

        execution_time = time.perf_counter() - t0
        print("Pipeline complete in {}", execution_time)

        # TODO: what do we need to return after pipeline ? 
        return result