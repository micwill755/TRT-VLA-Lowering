from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ExportPlan:
    """Single export plan for all stages. Stage-specific fields live in ``args``."""

    module: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    engine_dir: Path
    engine_file: str
    args: dict[str, Any] = field(default_factory=dict)
    trt_settings: dict[str, Any] = field(default_factory=dict)
    cleanup_modules: tuple[nn.Module, ...] = ()
    model_type: str = ""
    component: str = ""
