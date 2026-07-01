from __future__ import annotations

from trt.config.inference_config import HandleSource, InferenceRunConfig
from trt.context import EdgeContext


class VLAInferencePipeline:
    def __init__(self, config: InferenceRunConfig):
        self.config = config

    def run(self, ctx: EdgeContext) -> EdgeContext:
        if self.config.handle_source is HandleSource.IN_MEMORY:
            handles = ctx.handles.in_memory
        else:
            handles = ctx.handles.serialized
        self.config.hooks.run(ctx, handles)
        return ctx
