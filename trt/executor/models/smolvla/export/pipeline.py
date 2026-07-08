from trt.config.stage_config import PipelineConfig, StageConfig

_E = "trt.executor.models.smolvla.export"

SMOLVLA_PIPELINE = PipelineConfig(
    pipeline_name="SmolVLA",
    hooks={
        "preprocess": f"{_E}.process:preprocess",
        "postprocess": f"{_E}.process:postprocess",
    },
    stages=(
        StageConfig(
            stage_id=0,
            stage_name="vision",
            input_sources=(),
            runner="trt.runner.export:ExportRunner",
            engine_subdir="visual",
            hooks={
                "preprocess": f"{_E}.vision:preprocess",
                "export": f"{_E}.vision:export",
                "postprocess": f"{_E}.vision:postprocess",
            },
        ),
        StageConfig(
            stage_id=1,
            stage_name="language",
            input_sources=(0,),
            runner="trt.runner.export:ExportRunner",
            engine_subdir="language",
            hooks={
                "preprocess": f"{_E}.language:preprocess",
                "export": f"{_E}.language:export",
                "save_artifacts": f"{_E}.language:save_artifacts",
                "postprocess": f"{_E}.language:postprocess",
            },
        ),
        StageConfig(
            stage_id=2,
            stage_name="action",
            input_sources=(1,),
            runner="trt.runner.export:ExportRunner",
            engine_subdir="action",
            final_output=True,
            hooks={
                "preprocess": f"{_E}.diffusion:preprocess",
                "export": f"{_E}.diffusion:export",
                "postprocess": f"{_E}.diffusion:postprocess",
            },
        ),
    ),
)
