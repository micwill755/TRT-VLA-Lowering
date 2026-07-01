from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from trt.context import EdgeContext


@dataclass(frozen=True)
class BenchmarkStageConfig:
    name: str
    enabled: Callable[[EdgeContext], bool]
    run: Callable[[EdgeContext], None]


@dataclass(frozen=True)
class BenchmarkStageHooks:
    report: Callable[[EdgeContext], None] = field(default=lambda ctx: None)


@dataclass(frozen=True)
class BenchmarkPipelineConfig:
    backends: tuple[BenchmarkStageConfig, ...]
    hooks: BenchmarkStageHooks = field(default_factory=BenchmarkStageHooks)
