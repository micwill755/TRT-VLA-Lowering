from __future__ import annotations

import time

import torch

from trt.config.stage_config import PipelineConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve
from trt.inference.backends import InferenceBackend
from trt.inference.context import InferenceContext


class StagedInferencePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    @torch.no_grad()
    def run(
        self,
        edge_ctx: EdgeContext,
        backend: InferenceBackend,
        *,
        seed: int = 42,
    ) -> EdgeContext:
        if self.config.io is None:
            raise ValueError(f"{self.config.model_type} inference pipeline has no io spec")

        infer_ctx = InferenceContext(
            model=edge_ctx.model,
            policy=edge_ctx.policy,
            device=edge_ctx.device,
            model_inputs=edge_ctx.model_inputs,
            io=self.config.io,
            seed=seed,
        )
        if getattr(backend, "handles", None) is not None:
            infer_ctx.stage_handles = backend.handles

        torch.manual_seed(seed)
        if infer_ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        t0 = time.perf_counter()
        if self.config.hooks.preprocess:
            resolve(self.config.hooks.preprocess)(infer_ctx)

        for stage_cfg in self.config.stages:
            runner = resolve(stage_cfg.runner)(stage_cfg)
            runner.run(infer_ctx, backend)

        if self.config.hooks.postprocess:
            resolve(self.config.hooks.postprocess)(infer_ctx)

        edge_ctx.actions = infer_ctx.actions
        edge_ctx.export_state.setdefault("inference", {}).update(
            {
                "extras": {
                    "noise": infer_ctx.noise,
                    "visual_embeds": infer_ctx.image_embs,
                    "context_embs": infer_ctx.context_embs,
                    "stage_ms": dict(infer_ctx.stage_ms),
                },
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        return edge_ctx
