from trt.config.stage_config import PipelineConfig, StageConfig

_M = "trt.models.molmo2.export"

MOLMO2_PIPELINE = PipelineConfig(
    hooks={
        "preprocess": f"{_M}.preprocess:preprocess",
    },
    stages=(
        StageConfig(
            stage_id=0,
            input_sources=(),
            runner="trt.runner.export:ExportRunner",
            hooks={
                "plan_export": f"{_M}.vision:plan_export",
                "metadata":    f"{_M}.vision:metadata",
            },
        ),
    ),
)
