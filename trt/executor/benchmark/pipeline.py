from __future__ import annotations

from trt.config.benchmark_config import BenchmarkPipelineConfig, BenchmarkStageConfig, BenchmarkStageHooks
from trt.config.execution_mode import ExecutionMode
from trt.executor.benchmark.run import _has_in_memory, _has_serialized, _run_inference, report_action_parity

DEFAULT_BENCHMARK = BenchmarkPipelineConfig(
    backends=(
        BenchmarkStageConfig("pytorch", lambda ctx: True, lambda ctx: _run_inference(ctx, ExecutionMode.EAGER)),
        BenchmarkStageConfig(
            "in_memory_trt",
            _has_in_memory,
            lambda ctx: _run_inference(ctx, ExecutionMode.IN_MEMORY),
        ),
        BenchmarkStageConfig(
            "serialized_trt",
            _has_serialized,
            lambda ctx: _run_inference(ctx, ExecutionMode.SERIALIZED),
        ),
    ),
    hooks=BenchmarkStageHooks(report=report_action_parity),
)
