from __future__ import annotations

from typing import Union

from trt.hooks.export.action_context_plan import ActionContextExportPlan
from trt.hooks.export.action_plan import ActionExportPlan
from trt.hooks.export.base import ExportPlanBase
from trt.hooks.export.language_plan import LanguageExportPlan
from trt.hooks.export.vision_plan import VisionExportPlan

ExportPlanUnion = Union[
    VisionExportPlan,
    LanguageExportPlan,
    ActionContextExportPlan,
    ActionExportPlan,
]

__all__ = [
    "ActionContextExportPlan",
    "ActionExportPlan",
    "ExportPlanBase",
    "ExportPlanUnion",
    "LanguageExportPlan",
    "VisionExportPlan",
]
