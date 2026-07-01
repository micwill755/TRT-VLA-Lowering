from __future__ import annotations

from trt.config.stage_config import PipelineConfig
from trt.context import EdgeContext
from trt.export.mode import ExportMode
from trt.hooks.resolve import resolve
from trt.pipelines.legacy_export import run_legacy_export
from trt.runner.export import ExportRunner
from trt.profile import InMemoryHandles


class VLAExportPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config

    def run(self, ctx: EdgeContext, *, disk: bool = True, in_memory: bool = False) -> EdgeContext:
        if self.config is None or getattr(self.config, "use_legacy", False):
            return self._run_legacy(ctx, disk=disk, in_memory=in_memory)
        return self._run_configured(ctx)

    def _run_legacy(self, ctx: EdgeContext, *, disk: bool, in_memory: bool) -> EdgeContext:
        if disk:
            run_legacy_export(ctx, mode=ExportMode.SERIALIZED)
        if in_memory:
            handles = run_legacy_export(ctx, mode=ExportMode.IN_MEMORY)
            ctx.handles.in_memory = _merge_handles(ctx.handles.in_memory, handles)
        return ctx

    def _run_configured(self, ctx: EdgeContext) -> EdgeContext:
        assert self.config is not None
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


def _merge_handles(target: InMemoryHandles, exported: dict) -> InMemoryHandles:
    for key in ("vision", "language", "action", "action_context"):
        if key in exported and exported[key] is not None:
            setattr(target, key, exported[key])
    return target
