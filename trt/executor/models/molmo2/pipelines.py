# trt/models/molmo2/pipelines.py

from trt.config.stage_config import (
    PipelineConfig,
    PipelineHooks,
    StageConfig,
    StageHooks,
)

_M = "trt.models.molmo2.export"

MOLMO2_PIPELINE = PipelineConfig(
    model_type="MolmoAct2",
    hooks=PipelineHooks(
        preprocess=f"{_M}.preprocess:preprocess",
    ),
    stages=(
        StageConfig(
            stage_id=0,
            input_sources=(),
            runner="trt.runner.export:ExportRunner",
            hooks=StageHooks(
                plan_export=f"{_M}.vision:plan_export",
                metadata=f"{_M}.vision:metadata",
            ),
        ),
    ),
)
