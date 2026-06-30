# trt/executor/models/molmo2/pipeline.py

from trt.config.stage_config import (
    PipelineConfig,
    PipelineHooks,
    StageConfig,
    StageHooks,
    StageKind,
)

_M = "trt.executor.models.molmo2"

MOLMO2_PIPELINE = PipelineConfig(
    model_type="MolmoAct2",
    hooks=PipelineHooks(
        preprocess=f"{_M}.preprocess:preprocess",
    ),
    stages=(
        StageConfig(
            stage_id=0,
            kind=StageKind.VISION_ENCODE,
            input_sources=(),
            runner="trt.runner.export:ExportRunner",
            io=...,
            hooks=StageHooks(
                plan_export=f"{_M}.vision:plan_export",   # TokenPoolingExportModule
                metadata=f"{_M}.vision:metadata",
            ),
        ),
        # language + action hooks differ too, but runner is identical
        ...
    ),
)