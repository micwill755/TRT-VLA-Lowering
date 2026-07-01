from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class InMemoryHandles:
    vision: Any = None
    language: Any = None
    action: Any = None
    action_context: Any = None


@dataclass
class SerializedHandles:
    vision: Any = None
    language: Any = None
    action_context: Any = None
    action: Any = None


@dataclass
class ClonedLanguageSubgraph:
    """Cloned HF language stack for isolated TRT export."""

    language_model: nn.Module
    decoder: nn.Module
    lm_head: nn.Module
    config: Any
