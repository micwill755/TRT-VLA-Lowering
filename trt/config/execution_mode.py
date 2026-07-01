from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    EAGER = "eager"
    SERIALIZED = "serialized"
    IN_MEMORY = "in_memory"
