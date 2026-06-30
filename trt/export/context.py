"""Mutable state carried between VLA export pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.diffusion import DiffusionEngineSpec
from trt.io_spec import PipelineIOSpec
from trt.language import LanguageEngineSpec
from trt.vision import VisionEngineSpec


@dataclass
class ExportContext:
    model: nn.Module
    policy: Any
    device: torch.device
    model_inputs: dict[str, Any]
    io: PipelineIOSpec
    engine_root: Path | None = None
    seed: int = 42
    max_seq_len: int | None = None
    accuracy_check: bool = True

    pixel_values: torch.Tensor | None = None
    tokenized: dict[str, torch.Tensor] = field(default_factory=dict)
    action_side: dict[str, torch.Tensor] = field(default_factory=dict)

    vis_spec: VisionEngineSpec | None = None
    lang_spec: LanguageEngineSpec | None = None
    diffusion_spec: DiffusionEngineSpec | None = None
    image_embs: torch.Tensor | list[torch.Tensor] | None = None
    language_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    lm_hidden_states: torch.Tensor | None = None
    context_embs: torch.Tensor | None = None

    handles: dict[str, Any] = field(default_factory=dict)

    def engine_subdir(self, name: str) -> Path | None:
        if self.engine_root is None:
            return None
        return self.engine_root / name