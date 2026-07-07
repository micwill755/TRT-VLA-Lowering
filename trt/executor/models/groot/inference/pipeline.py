from trt.config.stage_config import (
    PipelineConfig,
    StageConfig
)
from trt.io_spec import GROOT_EDGE_IO

_I = "trt.executor.models.groot.inference"

GROOT_INFERENCE_PIPELINE = PipelineConfig(
    model_type="Gr00tN1d7",
    io=GROOT_EDGE_IO,
    hooks={
        preprocess=f"{_I}.process:preprocess",
        preprocess=f"{_I}.process:postprocess",
    },
    stages=(
        StageConfig(
            stage_id=0,
            stage_name="vision",
            input_sources=(),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="visual",
            hooks={
                preprocess=f"{_I}.vision:preprocess",
                execute=f"{_I}.vision:execute",
            },
        ),
        StageConfig(
            stage_id=1,
            stage_name="language",
            input_sources=(0,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="language",
            hooks={
                preprocess=f"{_I}.language:preprocess",
                run=f"{_I}.language:execute",
            },
        ),
        StageConfig(
            stage_id=2,
            stage_name="action_context",
            input_sources=(1,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="action_context",
            hooks={
                preprocess=f"{_I}.action_context:preprocess",
                run=f"{_I}.action_context:execute",
            },
        ),
        StageConfig(
            stage_id=3,
            stage_name="action",
            input_sources=(2,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="diffusion",
            final_output=True,
            hooks={
                preprocess=f"{_I}.diffusion:preprocess",
                run=f"{_I}.diffusion:execute",
            },
        ),
    ),
)
