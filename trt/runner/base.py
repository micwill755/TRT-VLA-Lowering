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
    handles: dict[str, Any] = field(default_factory=dict)

@dataclass
class StageResult:
    engine_path: Path | None = None
    spec: Any = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

class StageRunner(ABC):
    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult: ...