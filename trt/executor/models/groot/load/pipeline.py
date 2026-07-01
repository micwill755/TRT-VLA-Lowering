from trt.config.load_config import LoadPipelineConfig, SerializedStageSpec
from trt.serialize import (
    SerializedGrootAction,
    SerializedGrootActionContext,
    SerializedGrootLanguage,
    SerializedGrootVision,
)

GROOT_LOAD_PIPELINE = LoadPipelineConfig(
    stages=(
        SerializedStageSpec("vision", "visual", SerializedGrootVision),
        SerializedStageSpec("language", "language", SerializedGrootLanguage),
        SerializedStageSpec("action_context", "action_context", SerializedGrootActionContext),
        SerializedStageSpec("action", "action", SerializedGrootAction),
    ),
)
