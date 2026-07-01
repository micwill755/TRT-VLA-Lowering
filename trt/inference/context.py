"""Mutable state for VLA inference pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.io_spec import ComponentIOSpec, PipelineIOSpec


@dataclass
class LanguageOutputs:
    logits: torch.Tensor | None = None
    lm_hidden_states: torch.Tensor | None = None
    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None

    @classmethod
    def from_tuple(cls, outputs: tuple, io: ComponentIOSpec) -> LanguageOutputs:
        by_name = {
            io.output_names[i]: outputs[i]
            for i in range(min(len(io.output_names), len(outputs)))
        }
        return cls(
            logits=by_name.get("logits"),
            lm_hidden_states=by_name.get("lm_hidden_states"),
            prefix_k=by_name.get("prefix_k") or by_name.get("encoder_k"),
            prefix_v=by_name.get("prefix_v") or by_name.get("encoder_v"),
        )

    def as_tuple(self, io: ComponentIOSpec) -> tuple[torch.Tensor, ...]:
        values = []
        for name in io.output_names:
            value = getattr(self, name, None)
            if value is None:
                raise ValueError(f"Language output {name!r} is missing")
            values.append(value)
        return tuple(values)


@dataclass
class StageHandles:
    vision: Any = None
    language: Any = None
    action_context: Any = None
    action: Any = None


@dataclass
class InferenceContext:
    model: nn.Module
    policy: Any
    device: torch.device
    model_inputs: dict[str, Any]
    io: PipelineIOSpec
    seed: int = 42
    vision_module: nn.Module | None = None
    stage_handles: StageHandles | None = None

    pixel_values: torch.Tensor | None = None
    tokenized: dict[str, torch.Tensor] = field(default_factory=dict)
    action_side: dict[str, torch.Tensor] = field(default_factory=dict)

    image_embs: torch.Tensor | None = None
    language_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    lm: LanguageOutputs | None = None
    context_embs: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    actions: torch.Tensor | None = None

    extras: dict[str, Any] = field(default_factory=dict)
    stage_ms: dict[str, float] = field(default_factory=dict)
    stage_results: dict[int, Any] = field(default_factory=dict)
    engine_root: Path | None = None


@dataclass
class InferenceResult:
    actions: torch.Tensor
    extras: dict[str, Any]
    elapsed_s: float
    ctx: InferenceContext
