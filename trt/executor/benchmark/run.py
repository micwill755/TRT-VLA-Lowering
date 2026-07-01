from __future__ import annotations

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext

SEED = 42


def _run_inference(ctx: EdgeContext, mode: ExecutionMode) -> None:
    from trt.config.pipeline_registry import get_inference_pipeline
    from trt.pipelines.inference import InferencePipeline

    ctx.execution_mode = mode
    ctx.inference.seed = SEED
    ctx.stage_results.clear()
    model_type = getattr(ctx.profile, "pipeline_model_type", None) or ctx.profile.name
    InferencePipeline(get_inference_pipeline(model_type)).run(ctx)


def _has_in_memory(ctx: EdgeContext) -> bool:
    return ctx.handles.in_memory.vision is not None


def _has_serialized(ctx: EdgeContext) -> bool:
    return ctx.handles.serialized.vision is not None
