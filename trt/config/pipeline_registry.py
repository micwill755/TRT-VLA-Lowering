"""Registry mapping model types to export, inference, and benchmark pipelines.

Model packages register their stage configs at import time via
``register_*_pipeline``. Lookups use the profile ``name`` (e.g. ``gr00t``).
Used by :class:`EdgeOrchestrator` to pick the right pipeline for a
:class:`VLAProfile`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from trt.config.benchmark_config import BenchmarkPipelineConfig
from trt.config.stage_config import PipelineConfig
from trt.executor.models.groot.export.pipeline import GROOT_PIPELINE
from trt.executor.models.groot.inference.pipeline import GROOT_INFERENCE_PIPELINE
from trt.executor.benchmark.pipeline import DEFAULT_BENCHMARK

PipelineEntry: TypeAlias = PipelineConfig | Callable[[], PipelineConfig]

_EXPORT: dict[str, PipelineEntry] = {}
_INFERENCE: dict[str, PipelineEntry] = {}


def register_export_pipeline(model_type: str, config: PipelineEntry) -> None:
    """Register a TensorRT export pipeline for ``model_type``.

    ``config`` may be a :class:`PipelineConfig` or a zero-arg factory (for lazy
    init).
    """
    key = model_type.strip()
    if key in _EXPORT:
        raise KeyError(f"Export pipeline already registered for {key!r}")
    _EXPORT[key] = config


def register_inference_pipeline(model_type: str, config: PipelineEntry) -> None:
    """Register an inference-stage pipeline for ``model_type``."""
    key = model_type.strip()
    if key in _INFERENCE:
        raise KeyError(f"Inference pipeline already registered for {key!r}")
    _INFERENCE[key] = config


def _resolve(entry: PipelineEntry) -> PipelineConfig:
    """Return a concrete config, calling ``entry`` if it is a factory."""
    return entry() if callable(entry) else entry


def get_export_pipeline(model_type: str) -> PipelineConfig:
    """Look up the export pipeline for ``model_type``.

    Raises ``KeyError`` with known types if unregistered.
    """
    key = model_type.strip()
    if key not in _EXPORT:
        known = sorted(_EXPORT)
        raise KeyError(f"No export pipeline for {model_type!r}. Known: {known}")
    return _resolve(_EXPORT[key])


def get_inference_pipeline(model_type: str) -> PipelineConfig:
    """Look up the inference pipeline for ``model_type``."""
    key = model_type.strip()
    if key not in _INFERENCE:
        known = sorted(_INFERENCE)
        raise KeyError(f"No inference pipeline for {model_type!r}. Known: {known}")
    return _resolve(_INFERENCE[key])


def get_benchmark_pipeline(model_type: str | None = None) -> BenchmarkPipelineConfig:
    """Return the benchmark pipeline config.

    Currently model-agnostic; ``model_type`` is ignored and
    ``DEFAULT_BENCHMARK`` is always returned.
    """
    return DEFAULT_BENCHMARK


def get_pipeline_for_profile(profile) -> PipelineConfig:
    """Resolve the export pipeline from a profile instance."""
    name = getattr(profile, "name", None)
    if not name:
        raise ValueError(f"{type(profile).__name__} has no name")
    return get_export_pipeline(name)


def _register_builtin() -> None:
    """Register built-in model pipelines at module import time."""
    register_export_pipeline("gr00t", GROOT_PIPELINE)
    register_inference_pipeline("gr00t", GROOT_INFERENCE_PIPELINE)


_register_builtin()
