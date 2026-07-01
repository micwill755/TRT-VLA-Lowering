from __future__ import annotations

from trt.config.stage_config import PipelineConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve


class ExportPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, ctx: EdgeContext) -> EdgeContext:
        hooks = self.config.hooks
        if hooks.preprocess:
            resolve(hooks.preprocess)(ctx)
        for stage_cfg in self.config.stages:
            runner = resolve(stage_cfg.runner)(stage_cfg)
            result = runner.run(ctx)
            ctx.artifacts[f"stage_{stage_cfg.stage_id}"] = result
        if hooks.postprocess:
            resolve(hooks.postprocess)(ctx)
        return ctx
