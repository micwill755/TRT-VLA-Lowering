from __future__ import annotations

from enum import Enum


class ParityMode(str, Enum):
    """How benchmark stage parity is collected and reported.

    * ``e2e`` — each backend runs its own upstream outputs (honest deployment drift).
    * ``isolated`` — TRT stages receive eager reference upstream tensors.
    * ``both`` — run and report e2e and isolated passes.
    """

    E2E = "e2e"
    ISOLATED = "isolated"
    BOTH = "both"
