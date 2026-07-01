from __future__ import annotations

import time

import torch

from trt.config.stage_config import PipelineConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve


class InferencePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    @torch.no_grad()
    def run(self, ctx: EdgeContext) -> EdgeContext:
        if self.config.io is None:
            raise ValueError(f"{self.config.model_type} inference pipeline has no io spec")

        torch.manual_seed(ctx.inference.seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.inference.seed)

        t0 = time.perf_counter()
        hooks = self.config.hooks
        if hooks.preprocess:
            resolve(hooks.preprocess)(ctx)

        for stage_cfg in self.config.stages:
            runner = resolve(stage_cfg.runner)(stage_cfg)
            runner.run(ctx)

        if hooks.postprocess:
            resolve(hooks.postprocess)(ctx)

        for stage_cfg in self.config.stages:
            if stage_cfg.final_output:
                final = ctx.stage_results.get(stage_cfg.stage_id)
                if final is not None:
                    ctx.actions = final.tensors.get("actions")
                break

        ctx.export_state.setdefault("inference", {}).update(
            {
                "extras": {
                    "noise": ctx.inference.noise,
                    "visual_embeds": ctx.inference.image_embs,
                    "context_embs": ctx.inference.context_embs,
                    "stage_ms": dict(ctx.inference.stage_ms),
                },
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        return ctx
