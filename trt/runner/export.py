from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.hooks.resolve import resolve


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
        result = {**prepared, **exported}

        if hooks.get("save_artifacts"):
            resolve(hooks["save_artifacts"])(ctx, prepared, result)

        if hooks.get("postprocess"):
            result = resolve(hooks["postprocess"])(ctx, result)

        return result
