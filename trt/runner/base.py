# trt/runner/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

@dataclass
class StageContext:
    """Shared state passed between stages."""
    profile: Any
    policy: Any
    model: Any
    device: torch.device
    model_inputs: dict[str, torch.Tensor]
    engine_root: Path

    # filled by stages
    artifacts: dict[str, Any] = field(default_factory=dict)
    export_state: dict[str, Any] = field(default_factory=dict)
    handles: dict[str, Any] = field(default_factory=dict)

from trt.context import StageResult


class StageRunner(ABC):
    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult: ...