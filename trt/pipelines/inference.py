from __future__ import annotations

import time

import torch

from trt.context import EdgeContext
from trt.hooks.resolve import resolve


def _synchronize_if_cuda(ctx: EdgeContext) -> None:
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)


class InferencePipeline:
    def __init__(self, config):
        self.config = config
        self.hooks = config.hooks

    @torch.no_grad()
    def run(self, ctx: EdgeContext, inputs: dict | None = None) -> dict[int, dict]:
        torch.manual_seed(ctx.seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.seed)

        _synchronize_if_cuda(ctx)
        t0 = time.perf_counter()

        pipeline_inputs = dict(inputs or {})
        if self.hooks.get("preprocess"):
            prepared = resolve(self.hooks["preprocess"])(ctx) or {}
            pipeline_inputs.update(prepared)

        stage_outputs: dict[int, dict] = {}
        for stage_cfg in self.config.stages:
            if not ctx.benchmark_timing:
                print("Executing {}".format(stage_cfg.stage_name))
            # merge pipeline-level inputs with upstream stage output (linear DAG)
            upstream = (
                stage_outputs[stage_cfg.input_sources[0]]
                if stage_cfg.input_sources
                else {}
            )
            stage_inputs = {**pipeline_inputs, **upstream}
            runner = resolve(stage_cfg.runner)(stage_cfg)
            stg_output = runner.run(ctx, stage_inputs)
            stage_outputs[stage_cfg.stage_id] = stg_output

        if self.hooks.get("postprocess"):
            resolve(self.hooks["postprocess"])(ctx, stage_outputs)

        ctx.stage_results = stage_outputs
        _synchronize_if_cuda(ctx)
        execution_time = time.perf_counter() - t0
        if ctx.benchmark is not None and ctx.benchmark_timing:
            ctx.benchmark.record_e2e(ctx.execution_mode.value, execution_time)
        if not ctx.benchmark_timing:
            print(f"Pipeline complete in {execution_time:.2f}s")

        return stage_outputs
