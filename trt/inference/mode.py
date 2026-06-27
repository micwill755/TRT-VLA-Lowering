"""Inference mode and backend kind for VLA runtime tests."""

from __future__ import annotations

from enum import Enum, auto


class InferenceBackendKind(Enum):
    EAGER = auto()
    TRT_MODULE = auto()
    SERIALIZED = auto()


class InferenceMode(Enum):
    E2E = auto()
    STAGE_PARITY = auto()
