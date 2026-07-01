from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.profile import InMemoryHandles, SerializedHandles, VLAProfile


@dataclass
class StageResult:
    engine_path: Path | None = None
    spec: Any = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeHandles:
    in_memory: InMemoryHandles = field(default_factory=InMemoryHandles)
    serialized: SerializedHandles = field(default_factory=SerializedHandles)


@dataclass
class BenchmarkResult:
    timings: dict[str, list[float]] = field(default_factory=dict)

    def record(self, backend: str, seconds: float) -> None:
        self.timings.setdefault(backend, []).append(seconds)


@dataclass
class EdgeContext:
    profile: VLAProfile
    policy: Any
    model: nn.Module
    device: torch.device
    model_inputs: dict[str, Any]
    engine_root: Path
    args: Any

    artifacts: dict[str, StageResult] = field(default_factory=dict)
    export_state: dict[str, Any] = field(default_factory=dict)
    handles: EdgeHandles = field(default_factory=EdgeHandles)
    benchmark: BenchmarkResult | None = None
    actions: torch.Tensor | None = None
