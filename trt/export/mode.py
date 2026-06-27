"""Export mode for the VLA compile pipeline."""

from __future__ import annotations

from enum import Enum, auto


class ExportMode(Enum):
    """Where compiled components land after each pipeline stage."""

    IN_MEMORY = auto()
    SERIALIZED = auto()
