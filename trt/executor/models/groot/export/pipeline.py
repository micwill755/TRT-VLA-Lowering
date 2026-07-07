from trt.config.stage_config import PipelineConfig, StageConfig

_E = "trt.executor.models.groot.export"

GROOT_PIPELINE = PipelineConfig(
    pipeline_name="Gr00tN1d7",
    hooks={
        "preprocess":  f"{_E}.process:preprocess",
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
                "plan_export": f"{_E}.vision:plan_export",
                "compile":     f"{_E}.vision:compile",
                "metadata":    f"{_E}.vision:metadata",
            },
        ),
        StageConfig(
            stage_id=1,
            stage_name="language",
            input_sources=(0,),
            runner="trt.runner.export:ExportRunner",
            engine_subdir="language",
            hooks={
                "process_inputs": f"{_E}.glue:vision_to_language",
                "plan_export":    f"{_E}.language:plan_export",
                "compile":        f"{_E}.language:compile",
                "save_artifacts": f"{_E}.language:save_artifacts",
                "metadata":       f"{_E}.language:metadata",
            },
        ),
        StageConfig(
            stage_id=2,
            stage_name="action_context",
            input_sources=(1,),
            runner="trt.runner.export:ExportRunner",
            engine_subdir="action_context",
            hooks={
                "process_inputs": f"{_E}.glue:language_to_action_context",
                "plan_export":    f"{_E}.action_context:plan_export",
                "compile":        f"{_E}.action_context:compile",
                "metadata":       f"{_E}.action_context:metadata",
            },
        ),
        StageConfig(
            stage_id=3,
            stage_name="action",
            input_sources=(2,),
            runner="trt.runner.export:ExportRunner",
            engine_subdir="action",
            final_output=True,
            hooks={
                "process_inputs": f"{_E}.glue:action_context_to_action",
                "plan_export":    f"{_E}.action:plan_export",
                "compile":        f"{_E}.action:compile",
                "metadata":       f"{_E}.action:metadata",
            },
        ),
    ),
)
