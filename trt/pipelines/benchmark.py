from __future__ import annotations

import time

from trt.config.benchmark_config import BenchmarkPipelineConfig
from trt.config.inference_config import HandleSource, inference_config_for
from trt.config.pipeline_registry import get_eager_runner
from trt.context import BenchmarkResult, EdgeContext
from trt.measure import mean, print_timing
from trt.pipelines.inference import VLAInferencePipeline

SEED = 42


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


def _has_in_memory(ctx: EdgeContext) -> bool:
    return ctx.handles.in_memory.vision is not None


def _has_serialized(ctx: EdgeContext) -> bool:
    return ctx.handles.serialized.vision is not None


def default_groot_benchmark_config() -> BenchmarkPipelineConfig:
    from trt.config.benchmark_config import BackendConfig, BenchmarkHooks

    def run_eager(ctx: EdgeContext) -> None:
        model_type = getattr(ctx.profile, "pipeline_model_type", None) or ctx.profile.name
        try:
            get_eager_runner(model_type)(ctx)
        except KeyError:
            ctx.profile.run_inference_eager(
                ctx.model,
                ctx.policy,
                ctx.model_inputs,
                seed=SEED,
                device=ctx.device,
            )

    def run_in_memory(ctx: EdgeContext) -> None:
        VLAInferencePipeline(inference_config_for(ctx, HandleSource.IN_MEMORY)).run(ctx)

    def run_serialized(ctx: EdgeContext) -> None:
        VLAInferencePipeline(inference_config_for(ctx, HandleSource.SERIALIZED)).run(ctx)

    return BenchmarkPipelineConfig(
        backends=(
            BackendConfig("pytorch", lambda ctx: True, run_eager),
            BackendConfig("in_memory_trt", _has_in_memory, run_in_memory),
            BackendConfig("serialized_trt", _has_serialized, run_serialized),
        ),
        hooks=BenchmarkHooks(report=lambda ctx: None),
    )
