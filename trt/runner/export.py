from __future__ import annotations

from pathlib import Path

from trt.context import StageResult
from trt.hooks.export.plan import ExportPlan
from trt.hooks.resolve import resolve
from trt.runner.base import StageRunner
from trt.utils import free_cuda_memory


class ExportRunner(StageRunner):
    def __init__(self, stage_cfg):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    def run(self, ctx) -> StageResult:
        upstream = [ctx.artifacts[f"stage_{i}"] for i in self.stage_cfg.input_sources]

        stage_inputs = ctx.model_inputs
        if self.hooks.process_inputs:
            stage_inputs = resolve(self.hooks.process_inputs)(ctx, upstream, stage_inputs)

        if not self.hooks.plan_export:
            raise ValueError(
                f"export stage {self.stage_cfg.stage_id} missing plan_export hook"
            )
        plan: ExportPlan = resolve(self.hooks.plan_export)(ctx, stage_inputs)

        if not self.hooks.compile:
            raise ValueError(
                f"export stage {self.stage_cfg.stage_id} missing compile hook"
            )
        engine_path = resolve(self.hooks.compile)(plan)

        metadata: dict = {}
        if self.hooks.metadata:
            metadata = resolve(self.hooks.metadata)(ctx, plan)

        if self.hooks.save_artifacts:
            resolve(self.hooks.save_artifacts)(ctx, plan, engine_path)

        result = StageResult(
            engine_path=Path(engine_path),
            spec=plan,
            tensors=dict(plan.args.get("stage_tensors", {})),
            metadata=metadata,
        )

        if self.hooks.after_stage:
            resolve(self.hooks.after_stage)(ctx, result)

        for module in plan.cleanup_modules:
            free_cuda_memory(module)

        return result
