from __future__ import annotations

from pathlib import Path

from trt.context import StageResult
from trt.hooks.export.plan import ExportPlan
from trt.hooks.resolve import resolve
from trt.utils import free_cuda_memory


class ExportRunner:
    def __init__(self, stage_cfg):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    @torch.no_grad()
    def run(self, ctx: EdgeContext, inputs: dict) -> dict:
        hooks = self.hooks

        prepared = inputs
        if hooks.get("preprocess"):
            prepared = resolve(hooks["preprocess"])(ctx, inputs)

        exported = resolve(hooks["export"])(ctx, prepared)
        prepared.update(exported)

        if hooks.get("postprocess"):
            result = resolve(hooks["postprocess"])(ctx, result)

        return result
