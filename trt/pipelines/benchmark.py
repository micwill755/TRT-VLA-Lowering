from __future__ import annotations

import time

from trt.config.benchmark_config import BenchmarkPipelineConfig
from trt.context import BenchmarkResult, EdgeContext
from trt.measure import mean, print_timing


class BenchmarkPipeline:
    def __init__(self, config: BenchmarkPipelineConfig):
        self.config = config

    def run(self, ctx: EdgeContext) -> BenchmarkResult:
        result = BenchmarkResult()
        warmup = int(getattr(ctx.args, "warmup", 3))
        iterations = int(getattr(ctx.args, "num_iterations", 12))

        for _ in range(iterations):
            for backend in self.config.backends:
                if not backend.enabled(ctx):
                    continue
                t0 = time.perf_counter()
                backend.run(ctx)
                result.record(backend.name, time.perf_counter() - t0)

        ctx.benchmark = result
        self.config.hooks.report(ctx)
        for name, samples in result.timings.items():
            if len(samples) > warmup:
                print_timing(name, mean(samples[warmup:]))
        return result
