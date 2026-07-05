from __future__ import annotations

import time

from trt.config.benchmark_config import BenchmarkPipelineConfig
from trt.context import BenchmarkResult, EdgeContext
from trt.measure import print_timing

class BenchmarkPipeline:
    def __init__(self, config: BenchmarkPipelineConfig):
        self.config = config

    def run(self, ctx: EdgeContext) -> BenchmarkResult:
        if hooks.preprocess:
            resolve(hooks.preprocess)(ctx)
        
        for stage_cfg in self.config.stages:
            print("Executing {}".format(stage_cfg.name))
            stg_input = stg_output
            runner = resolve(stage_cfg.runner)(stage_cfg)
            stg_output = runner.run(ctx, stg_input)
        
        if hooks.postprocess:
            resolve(hooks.postprocess)(ctx)
        
        '''result = BenchmarkResult()
        warmup = int(getattr(ctx.args, "warmup", 3))
        iterations = int(getattr(ctx.args, "num_iterations", 12))

        for iter_idx in range(iterations):
            for stage in self.config.stages:
                if not stage.enabled(ctx):
                    continue
                t0 = time.perf_counter()
                stage.run(ctx)
                result.record(stage.name, time.perf_counter() - t0)
                if iter_idx == iterations - 1:
                    if ctx.actions is not None:
                        result.record_actions(stage.name, ctx.actions)
                    if ctx.inference.image_embs is not None:
                        result.record_image_embs(stage.name, ctx.inference.image_embs)

        ctx.benchmark = result
        for name, samples in result.timings.items():
            if len(samples) > warmup:
                print_timing(name, [s * 1000 for s in samples[warmup:]])
        self.config.hooks.report(ctx)
        return result'''
        pass

