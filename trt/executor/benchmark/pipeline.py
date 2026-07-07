from __future__ import annotations

from trt.config.stage_config import PipelineConfig, StageConfig

_I = "trt.executor.benchmark"

DEFAULT_BENCHMARK = PipelineConfig(
    hooks={
        "preprocess": f"{_I}.preprocess:preprocess",
    },
    stages=(
        StageConfig(
            stage_id=0,
            stage_name="eager",
            input_sources=(),
            runner="trt.runner.benchmark:BenchmarkRunner",
            hooks={
                "preprocess": f"{_I}.eager:preprocess",
                "execute":    f"{_I}.eager:execute",
            },
        ),
        StageConfig(
            stage_id=1,
            stage_name="in_memory",
            input_sources=(0,),
            runner="trt.runner.benchmark:BenchmarkRunner",
            engine_subdir="compiled",
            hooks={
                "process_inputs": f"{_I}.in_memory:eager_to_in_memory",
                "execute":        f"{_I}.in_memory:execute",
            },
        ),
        StageConfig(
            stage_id=2,
            stage_name="serialized",
            input_sources=(1,),
            runner="trt.runner.benchmark:BenchmarkRunner",
            hooks={
                "process_inputs": f"{_I}.glue:in_memory_to_serialized",
                "execute":        f"{_I}.serialized:execute",
            },
        )
    ),
)
