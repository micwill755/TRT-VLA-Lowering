from __future__ import annotations

from pathlib import Path

from trt.export_compile.action import compile_action_plan
from trt.export_compile.action_context import compile_action_context_plan
from trt.export_compile.language import compile_language_plan
from trt.export_compile.vision import compile_vision_plan
from trt.hooks.export import (
    ActionContextExportPlan,
    ActionExportPlan,
    ExportPlanUnion,
    LanguageExportPlan,
    VisionExportPlan,
)


def compile_export_plan(plan: ExportPlanUnion) -> Path:
    if isinstance(plan, VisionExportPlan):
        return compile_vision_plan(plan)
    if isinstance(plan, LanguageExportPlan):
        return compile_language_plan(plan)
    if isinstance(plan, ActionContextExportPlan):
        return compile_action_context_plan(plan)
    if isinstance(plan, ActionExportPlan):
        return compile_action_plan(plan)
    raise TypeError(f"Unsupported export plan type: {type(plan).__name__}")


__all__ = [
    "compile_action_context_plan",
    "compile_action_plan",
    "compile_export_plan",
    "compile_language_plan",
    "compile_vision_plan",
]
