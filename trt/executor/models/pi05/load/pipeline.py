from trt.config.load_config import LoadPipelineConfig, SerializedStageSpec
from trt.executor.models.pi05.load.serialize import (
    SerializedPI05Action,
    SerializedPI05Language,
    SerializedPI05Vision,
)

PI05_LOAD_PIPELINE = LoadPipelineConfig(
    stages=(
        SerializedStageSpec("vision", "visual", SerializedPI05Vision),
        SerializedStageSpec("language", "language", SerializedPI05Language),
        SerializedStageSpec("action", "action", SerializedPI05Action),
    ),
)
