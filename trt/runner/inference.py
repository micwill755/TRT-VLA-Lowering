from __future__ import annotations

import time

import torch

from trt.config.execution_mode import ExecutionMode
from trt.config.stage_config import StageConfig
from trt.context import EdgeContext
from trt.hooks.resolve import resolve


def _synchronize_if_cuda(ctx: EdgeContext) -> None:
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)


class InferenceRunner:
    def __init__(self, stage_cfg: StageConfig):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    def _cache_key(self, ctx: EdgeContext) -> str:
        return f"{self.stage_cfg.stage_id}:{ctx.execution_mode.value}"

    @torch.no_grad()
    def run(self, ctx: EdgeContext, inputs: dict) -> dict:
        hooks = self.hooks
        cache_key = self._cache_key(ctx)

        if cache_key in ctx.stage_execute_cache:
            prepared = ctx.stage_execute_cache[cache_key]
        else:
            prepared = inputs
            if hooks.get("preprocess"):
                prepared = resolve(hooks["preprocess"])(ctx, inputs)

            match ctx.execution_mode:
                case ExecutionMode.IN_MEMORY:
                    if hooks.get("compile"):
                        compiled = resolve(hooks["compile"])(ctx, prepared)
                        if compiled:
                            prepared.update(compiled)

                case ExecutionMode.SERIALIZED:
                    if hooks.get("load"):
                        loaded = resolve(hooks["load"])(ctx, prepared)
                        if loaded:
                            prepared.update(loaded)

                case ExecutionMode.EAGER:
                    pass

            ctx.stage_execute_cache[cache_key] = prepared

        _synchronize_if_cuda(ctx)
        t0 = time.perf_counter()
        result = resolve(hooks["execute"])(ctx, prepared)
        _synchronize_if_cuda(ctx)
        elapsed_s = time.perf_counter() - t0

        if hooks.get("postprocess"):
            result = resolve(hooks["postprocess"])(ctx, result)

        metadata = dict(result.get("metadata", {}))
        metadata["elapsed_s"] = elapsed_s
        result["metadata"] = metadata

        if ctx.benchmark is not None and ctx.benchmark_timing:
            ctx.benchmark.record_stage_execute(
                ctx.execution_mode.value,
                self.stage_cfg.stage_name,
                elapsed_s,
            )

        return result
