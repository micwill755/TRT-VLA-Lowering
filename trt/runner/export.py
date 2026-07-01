from __future__ import annotations

from pathlib import Path

import torch

from trt.context import StageResult
from trt.export_compile import compile_export_plan
from trt.hooks.export import (
    ExportPlanUnion,
    LanguageExportPlan,
    VisionExportPlan,
)
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
                f"export stage {self.stage_cfg.stage_id} ({self.stage_cfg.kind}) missing plan_export hook"
            )
        plan: ExportPlanUnion = resolve(self.hooks.plan_export)(ctx, stage_inputs)

        with torch.no_grad():
            eager_output = plan.module(*plan.sample_inputs)

        engine_path = compile_export_plan(plan)

        metadata: dict = {}
        if self.hooks.metadata:
            metadata = resolve(self.hooks.metadata)(ctx, plan, eager_output)

        if self.hooks.save_artifacts:
            resolve(self.hooks.save_artifacts)(ctx, plan, engine_path)

        result = StageResult(
            engine_path=Path(engine_path),
            spec=plan if isinstance(plan, LanguageExportPlan) else None,
            tensors=self._named_outputs(plan, eager_output),
            metadata=metadata,
        )

        if self.hooks.after_stage:
            resolve(self.hooks.after_stage)(ctx, result)

        for module in plan.cleanup_modules:
            free_cuda_memory(module)

        return result

    @staticmethod
    def _named_outputs(plan: ExportPlanUnion, output) -> dict[str, torch.Tensor]:
        names = plan.output_names
        if isinstance(output, (tuple, list)):
            tensors = {name: value for name, value in zip(names, output)}
        elif len(names) == 1:
            tensors = {names[0]: output}
        else:
            tensors = {"output": output}

        if isinstance(plan, VisionExportPlan):
            if "image_embeddings" in tensors and "image_embs" not in tensors:
                tensors["image_embs"] = tensors["image_embeddings"]
        if isinstance(plan, LanguageExportPlan):
            if "lm_hidden_states" in tensors and "hidden_states" not in tensors:
                tensors["hidden_states"] = tensors["lm_hidden_states"]
            elif isinstance(output, (tuple, list)) and len(output) > 1 and "hidden_states" not in tensors:
                tensors["hidden_states"] = output[1]
        return tensors
