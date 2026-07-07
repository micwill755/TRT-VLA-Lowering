from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    EAGER = "eager"
    IN_MEMORY = "in_memory"
    SERIALIZED = "serialized"
