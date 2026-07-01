from __future__ import annotations

from trt.config.load_config import LoadPipelineConfig
from trt.config.pipeline_registry import get_load_pipeline
from trt.context import EdgeContext
from trt.serialize import load_serialized_modules
from trt.profile import SerializedHandles


class LoadPipeline:
    def __init__(self, config: LoadPipelineConfig):
        self.config = config

    @classmethod
    def for_model_type(cls, model_type: str) -> LoadPipeline:
        return cls(get_load_pipeline(model_type))

    def run(self, ctx: EdgeContext) -> EdgeContext:
        if not self.config.stages:
            ctx.handles.serialized = SerializedHandles()
            return ctx
        specs = tuple(stage.to_module_spec() for stage in self.config.stages)
        loaded = load_serialized_modules(ctx.engine_root, specs=specs)
        handles = SerializedHandles()
        for index, stage in enumerate(self.config.stages):
            setattr(handles, stage.key, loaded[index])
        ctx.handles.serialized = handles
        return ctx
