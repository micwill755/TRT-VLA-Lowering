# trt/executor/models/groot/pipeline.py

from trt.config.stage_config import (
    PipelineConfig,
    PipelineHooks,
    StageConfig,
    StageHooks,
    StageKind,
)
from trt.io_spec import GROOT_EDGE_IO

_H = "trt.executor.models.gr00t"

GROOT_PIPELINE = PipelineConfig(
    model_type="Gr00tN1d7",
    hooks=PipelineHooks(
        preprocess=f"{_H}.preprocess:preprocess",
        postprocess=f"{_H}.postprocess:postprocess",
    ),
    stages=(
        StageConfig(
            stage_id=0,
            kind=StageKind.VISION_ENCODE,
            input_sources=(),
            runner="trt.runner.export:ExportRunner",
            io=GROOT_EDGE_IO.vision,
            engine_subdir="visual",
            hooks=StageHooks(
                plan_export=f"{_H}.vision:plan_export",
                metadata=f"{_H}.vision:metadata",
            ),
        ),
        StageConfig(
            stage_id=1,
            kind=StageKind.LANGUAGE_PREFILL,
            input_sources=(0,),
            runner="trt.runner.export:ExportRunner",
            io=GROOT_EDGE_IO.language,
            engine_subdir="language",
            hooks=StageHooks(
                process_inputs=f"{_H}.glue:vision_to_language",
                plan_export=f"{_H}.language:plan_export",
                save_artifacts=f"{_H}.language:save_artifacts",
                metadata=f"{_H}.language:metadata",
            ),
        ),
        StageConfig(
            stage_id=2,
            kind=StageKind.ACTION_ROLLOUT,
            input_sources=(1, 0),   # LM hidden + vision metadata if needed
            runner="trt.runner.export:ExportRunner",
            io=GROOT_EDGE_IO.action,
            engine_subdir="action",
            final_output=True,
            hooks=StageHooks(
                process_inputs=f"{_H}.glue:language_to_action",
                plan_export=f"{_H}.action:plan_export",
            ),
        ),
    ),
)