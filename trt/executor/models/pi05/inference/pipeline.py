from trt.config.stage_config import PipelineConfig, StageConfig

_I = "trt.executor.models.groot.inference"

# stage_name -> tensor key used for eager vs TRT/serialized parity checks.
STAGE_PARITY_TENSORS = {
    "vision": "image_embs",
    "language": "lm_hidden",
    "action_context": "context_embs",
    "action": "actions",
}

GROOT_INFERENCE_PIPELINE = PipelineConfig(
    pipeline_name="Gr00tN1d7",
    hooks={
        "preprocess":  f"{_I}.process:preprocess",
        "postprocess": f"{_I}.process:postprocess",
    },
    stages=(
        StageConfig(
            stage_id=0,
            stage_name="vision",
            input_sources=(),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="visual",
            hooks={
                "preprocess":  f"{_I}.vision:preprocess",
                "compile":     f"{_I}.vision:compile",
                "load":        f"{_I}.vision:load",
                "execute":     f"{_I}.vision:execute",
                "postprocess": f"{_I}.vision:postprocess",
            },
        ),
        StageConfig(
            stage_id=1,
            stage_name="language",
            input_sources=(0,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="language",
            hooks={
                "preprocess":  f"{_I}.language:preprocess",
                "compile":     f"{_I}.language:compile",
                "load":        f"{_I}.language:load",
                "execute":     f"{_I}.language:execute",
                "postprocess": f"{_I}.language:postprocess",
            },
        ),
        StageConfig(
            stage_id=2,
            stage_name="action_context",
            input_sources=(1,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="action_context",
            hooks={
                "preprocess":  f"{_I}.action_context:preprocess",
                "compile":     f"{_I}.action_context:compile",
                "load":        f"{_I}.action_context:load",
                "execute":     f"{_I}.action_context:execute",
                "postprocess": f"{_I}.action_context:postprocess",
            },
        ),
        StageConfig(
            stage_id=3,
            stage_name="action",
            input_sources=(2,),
            runner="trt.runner.inference:InferenceRunner",
            engine_subdir="diffusion",
            final_output=True,
            hooks={
                "preprocess":  f"{_I}.diffusion:preprocess",
                "compile":     f"{_I}.diffusion:compile",
                "load":        f"{_I}.diffusion:load",
                "execute":     f"{_I}.diffusion:execute",
                "postprocess": f"{_I}.diffusion:postprocess",
            },
        ),
    ),
)
