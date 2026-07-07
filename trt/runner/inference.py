from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from trt.config.stage_config import StageConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve

class InferenceRunner:
    def __init__(self, stage_cfg: StageConfig):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    @torch.no_grad()
    def run(self, ctx, inputs):
        hooks = self.hooks  # dict

        prepared = inputs
        if hooks.get("preprocess"):
            prepared = resolve(hooks["preprocess"])(ctx, inputs)

        if hooks.get("compile"):
            compiled = resolve(hooks["compile"])(ctx, prepared)
            if compiled:
                prepared.update(compiled)

        if hooks.get("load"):
            loaded = resolve(hooks["load"])(ctx, prepared)
            if loaded:
                prepared.update(loaded)

        result = resolve(hooks["execute"])(ctx, prepared)

        if hooks.get("postprocess"):
            result = resolve(hooks["postprocess"])(ctx, result)

        return result
