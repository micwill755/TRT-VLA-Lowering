from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from trt.config.pipeline_registry import get_inference_pipeline
from trt.context import EdgeContext
from trt.profile import InMemoryHandles, SerializedHandles


class HandleSource(str, Enum):
    IN_MEMORY = "in_memory"
    SERIALIZED = "serialized"


@dataclass(frozen=True)
class InferenceRunHooks:
    run: Callable[[EdgeContext, InMemoryHandles | SerializedHandles], None]


@dataclass(frozen=True)
class InferenceRunConfig:
    handle_source: HandleSource
    hooks: InferenceRunHooks


def default_inference_config(handle_source: HandleSource) -> InferenceRunConfig:
    def _run(ctx: EdgeContext, handles: InMemoryHandles | SerializedHandles) -> None:
        ctx.profile.run_inference_trt(
            ctx.model,
            ctx.policy,
            ctx.model_inputs,
            handles=handles,
            seed=42,
            device=ctx.device,
        )

    return InferenceRunConfig(handle_source=handle_source, hooks=InferenceRunHooks(run=_run))


def inference_config_for(ctx: EdgeContext, handle_source: HandleSource) -> InferenceRunConfig:
    model_type = getattr(ctx.profile, "pipeline_model_type", None) or ctx.profile.name
    try:
        return get_inference_pipeline(model_type, handle_source)
    except KeyError:
        return default_inference_config(handle_source)
