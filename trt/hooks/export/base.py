from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn


@dataclass
class ExportPlanBase:
    """Shared compile destination fields for all staged export plans."""

    engine_dir: Path
    engine_file: str
    trt_settings: dict[str, Any] = field(default_factory=dict)
    cleanup_modules: tuple[nn.Module, ...] = ()
    model_type: str = ""
    component: str = ""
