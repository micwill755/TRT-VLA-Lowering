from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from trt.context import EdgeContext


@dataclass(frozen=True)
class BackendConfig:
    name: str
    enabled: Callable[[EdgeContext], bool]
    run: Callable[[EdgeContext], None]


@dataclass(frozen=True)
class BenchmarkHooks:
    report: Callable[[EdgeContext], None] = field(default=lambda ctx: None)


@dataclass(frozen=True)
class BenchmarkPipelineConfig:
    backends: tuple[BackendConfig, ...]
    hooks: BenchmarkHooks = field(default_factory=BenchmarkHooks)
