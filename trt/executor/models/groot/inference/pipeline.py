from trt.config.stage_config import (
    PipelineConfig,
    PipelineHooks,
    StageConfig,
    StageHooks,
)
from trt.io_spec import GROOT_EDGE_IO

_I = "trt.executor.models.groot.inference"

GROOT_INFERENCE_PIPELINE = PipelineConfig(
    model_type="Gr00tN1d7",
    io=GROOT_EDGE_IO,
    hooks=PipelineHooks(
        preprocess=f"{_I}.preprocess:preprocess",
    ),
    stages=(
        StageConfig(
            stage_id=0,
            input_sources=(),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="visual",
            hooks=StageHooks(
                run=f"{_I}.vision:run",
            ),
        ),
        StageConfig(
            stage_id=1,
            input_sources=(0,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="language",
            hooks=StageHooks(
                process_inputs=f"{_I}.glue:vision_to_language",
                run=f"{_I}.language:run",
            ),
        ),
        StageConfig(
            stage_id=2,
            input_sources=(1,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="action_context",
            hooks=StageHooks(
                process_inputs=f"{_I}.glue:language_to_action_context",
                run=f"{_I}.action_context:run",
            ),
        ),
        StageConfig(
            stage_id=3,
            input_sources=(2,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="action",
            final_output=True,
            hooks=StageHooks(
                run=f"{_I}.action:run",
            ),
        ),
    ),
)
