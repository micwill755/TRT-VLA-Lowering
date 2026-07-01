from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.config.pipeline_registry import get_inference_pipeline_config
from trt.inference.backends import EagerBackend, TrtModuleBackend, stage_handles_from_modules
from trt.pipelines.inference_staged import StagedInferencePipeline
from trt.profile import InMemoryHandles, SerializedHandles

SEED = 42


def _pipeline_config(ctx: EdgeContext):
    model_type = getattr(ctx.profile, "pipeline_model_type", None) or ctx.profile.name
    return get_inference_pipeline_config(model_type)


@torch.no_grad()
def run_eager(ctx: EdgeContext) -> None:
    StagedInferencePipeline(_pipeline_config(ctx)).run(ctx, EagerBackend(), seed=SEED)


@torch.no_grad()
def run_trt(ctx: EdgeContext, handles: InMemoryHandles | SerializedHandles) -> None:
    backend = TrtModuleBackend(
        stage_handles_from_modules(
            vision=handles.vision,
            language=handles.language,
            action_context=getattr(handles, "action_context", None),
            action=handles.action,
        )
    )
    StagedInferencePipeline(_pipeline_config(ctx)).run(ctx, backend, seed=SEED)
