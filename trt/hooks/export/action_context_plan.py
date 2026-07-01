from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from trt.hooks.export.base import ExportPlanBase


@dataclass
class ActionContextExportPlan(ExportPlanBase):
    module: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    extra_config: dict[str, Any] = field(default_factory=dict)
