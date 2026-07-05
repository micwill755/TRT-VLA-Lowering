"""Shared run-state types for Edge-LLM export, load, and benchmark pipelines.

A single :class:`EdgeContext` is created per orchestrator run and passed through
export, load, and benchmark stages. Other dataclasses here hold nested slices of
that state (inference scratch space, stage outputs, TRT handles, timings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.config.execution_mode import ExecutionMode
from trt.profile import InMemoryHandles, SerializedHandles, VLAProfile



@dataclass
class InferenceState:
    """Mutable scratch space for one end-to-end inference pass.

    Vision, language, and action stages read/write here as they run inside
    :class:`EdgeContext`. Populated during benchmark; also used when comparing
    eager vs TRT backends on the same prepared inputs.
    """

    seed: int = 42
    tokenized: dict[str, torch.Tensor] = field(default_factory=dict)
    pixel_values: torch.Tensor | None = None
    image_embs: torch.Tensor | None = None
    language_inputs: dict[str, Any] = field(default_factory=dict)
    lm_hidden_states: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None
    context_embs: torch.Tensor | None = None
    action_side: dict[str, Any] = field(default_factory=dict)
    noise: torch.Tensor | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class StageResult:
    """Output of one export stage (vision, language, or action).

    Stored in ``EdgeContext.artifacts`` under keys like ``stage_0``. Holds the
    compiled engine path, IO spec, representative tensors, and any stage metadata
    needed by later stages or load/benchmark.
    """

    engine_path: Path | None = None
    spec: Any = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeHandles:
    """Loaded TRT runners for in-memory and serialized execution.

    Export may populate ``in_memory`` handles for same-process TRT runs.
    :class:`LoadPipeline` fills ``serialized`` from ``engine_root`` for
    benchmark and parity checks against eager PyTorch.
    """

    in_memory: InMemoryHandles = field(default_factory=InMemoryHandles)
    serialized: SerializedHandles = field(default_factory=SerializedHandles)


@dataclass
class BenchmarkResult:
    """Timing and action outputs collected across benchmark stages.

    Each stage (eager, TRT in-memory, serialized engines, etc.) appends
    latencies and optional predicted actions for parity reporting.
    """

    timings: dict[str, list[float]] = field(default_factory=dict)
    actions: dict[str, torch.Tensor] = field(default_factory=dict)
    image_embs: dict[str, torch.Tensor] = field(default_factory=dict)

    def record(self, backend: str, seconds: float) -> None:
        self.timings.setdefault(backend, []).append(seconds)

    def record_actions(self, backend: str, actions: torch.Tensor) -> None:
        self.actions[backend] = actions.detach()

    def record_image_embs(self, backend: str, image_embs: torch.Tensor) -> None:
        self.image_embs[backend] = image_embs.detach()


@dataclass
class EdgeContext:
    """Shared session object for a single Edge-LLM export/benchmark run.

    Created once by the orchestrator with the loaded profile, policy, model, one
    prepared sample batch (``model_inputs``), and CLI args. Passed through
    export, load, and benchmark pipelines; stages read inputs from it and write
    intermediate results back (``export_state``, ``artifacts``, ``handles``,
    ``inference``, ``benchmark``).

    Flow::

        EdgeOrchestrator._build_context()
                │
                ▼
           EdgeContext ──► ExportPipeline  → artifacts, export_state
                │
                ├──► LoadPipeline         → handles.serialized
                │
                └──► BenchmarkPipeline    → inference, benchmark, actions
    """

    # Set at creation: model setup + one compile/benchmark sample.
    profile: VLAProfile
    policy: Any
    model: nn.Module
    device: torch.device
    dtype: Any
    model_inputs: dict[str, Any]
    engine_root: Path
    args: Any
    seed: int = 42

    # Filled as pipelines and stages run.
    artifacts: dict[str, StageResult] = field(default_factory=dict)
    export_state: dict[str, Any] = field(default_factory=dict)
    handles: EdgeHandles = field(default_factory=EdgeHandles)
    benchmark: BenchmarkResult | None = None
    actions: torch.Tensor | None = None

    execution_mode: ExecutionMode = ExecutionMode.EAGER
    inference: InferenceState = field(default_factory=InferenceState)
    stage_results: dict[int, Any] = field(default_factory=dict)
