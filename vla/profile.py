"""Backward-compatible re-exports. Prefer ``trt.profile``."""

from trt.profile import (
    ClonedLanguageSubgraph,
    InMemoryHandles,
    SerializedHandles,
    VLAProfile,
)

__all__ = [
    "ClonedLanguageSubgraph",
    "InMemoryHandles",
    "SerializedHandles",
    "VLAProfile",
]
