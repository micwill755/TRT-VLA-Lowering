# trt/hooks/export_plan.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

@dataclass
class ExportPlan:
    module: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    engine_dir: Path
    engine_file: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    extra_config: dict[str, Any] = field(default_factory=dict)
    trt_settings: dict[str, Any] = field(default_factory=dict)
    patch_target: nn.Module | None = None
    patch_batch_size: int = 0
    patch_seq_len: int = 0
    cleanup_modules: tuple[nn.Module, ...] = ()