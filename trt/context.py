from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.config.execution_mode import ExecutionMode
from trt.profile import InMemoryHandles, SerializedHandles, VLAProfile


@dataclass
class LanguageOutputs:
    lm_hidden_states: torch.Tensor | None = None
    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None


@dataclass
class InferenceState:
    seed: int = 42
    tokenized: dict[str, torch.Tensor] = field(default_factory=dict)
    pixel_values: torch.Tensor | None = None
    image_embs: torch.Tensor | None = None
    language_inputs: dict[str, Any] = field(default_factory=dict)
    lm: LanguageOutputs | None = None
    context_embs: torch.Tensor | None = None
    action_side: dict[str, Any] = field(default_factory=dict)
    noise: torch.Tensor | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)


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

    execution_mode: ExecutionMode = ExecutionMode.EAGER
    inference: InferenceState = field(default_factory=InferenceState)
    stage_results: dict[int, Any] = field(default_factory=dict)
