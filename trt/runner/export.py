# trt/runner/export.py

from trt.hooks.resolve import resolve
from trt.runner.base import StageContext, StageResult, StageRunner

class ExportRunner(StageRunner):
    def __init__(self, stage_cfg):
        self.stage_cfg = stage_cfg
        self.hooks = stage_cfg.hooks

    def run(self, ctx: StageContext) -> StageResult:
        upstream = [ctx.artifacts[f"stage_{i}"] for i in self.stage_cfg.input_sources]

        stage_inputs = ctx.model_inputs
        if self.hooks.process_inputs:
            stage_inputs = resolve(self.hooks.process_inputs)(ctx, upstream, stage_inputs)

        plan = resolve(self.hooks.plan_export)(ctx, stage_inputs)

        with torch.no_grad():
            eager_output = plan.module(*plan.sample_inputs)

        # patch → save_trt_engine_module → restore → free
        engine_path = ...  # same as your GridVisionExportRunner today

        metadata = {}
        if self.hooks.metadata:
            metadata = resolve(self.hooks.metadata)(ctx, plan, eager_output)

        if self.hooks.save_artifacts:
            resolve(self.hooks.save_artifacts)(ctx, plan, engine_path)

        result = StageResult(
            engine_path=engine_path,
            tensors=self._named_outputs(plan, eager_output),
            metadata=metadata,
        )

        if self.hooks.after_stage:
            resolve(self.hooks.after_stage)(ctx, result)

        return result