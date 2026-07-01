from trt.config.stage_config import (
    PipelineConfig,
    PipelineHooks,
    StageConfig,
    StageHooks,
    StageKind,
)
from trt.io_spec import GROOT_EDGE_IO
from trt.executor.models.groot.load.pipeline import GROOT_LOAD_PIPELINE

_E = "trt.executor.models.groot.export"

GROOT_PIPELINE = PipelineConfig(
    model_type="Gr00tN1d7",
    use_legacy=True,
    hooks=PipelineHooks(
        preprocess=f"{_E}.preprocess:preprocess",
        postprocess=f"{_E}.postprocess:postprocess",
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
                plan_export=f"{_E}.vision:plan_export",
                metadata=f"{_E}.vision:metadata",
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
                process_inputs=f"{_E}.glue:vision_to_language",
                plan_export=f"{_E}.language:plan_export",
                save_artifacts=f"{_E}.language:save_artifacts",
                metadata=f"{_E}.language:metadata",
            ),
        ),
        StageConfig(
            stage_id=2,
            kind=StageKind.ACTION_ROLLOUT,
            input_sources=(1, 0),
            runner="trt.runner.export:ExportRunner",
            io=GROOT_EDGE_IO.action,
            engine_subdir="action",
            final_output=True,
            hooks=StageHooks(
                process_inputs=f"{_E}.glue:language_to_action",
                plan_export=f"{_E}.action:plan_export",
            ),
        ),
    ),
)
