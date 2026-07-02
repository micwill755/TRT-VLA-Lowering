"""Registry mapping model types to export, inference, load, and benchmark pipelines.

Model packages register their stage configs at import time via
``register_*_pipeline``. Lookups accept canonical names (e.g. ``Gr00tN1d7``) or
aliases (e.g. ``gr00t``, ``groot``). Used by :class:`EdgeOrchestrator` to pick
the right pipeline for a :class:`VLAProfile`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from trt.config.benchmark_config import BenchmarkPipelineConfig
from trt.config.load_config import LoadPipelineConfig
from trt.config.stage_config import PipelineConfig
from trt.executor.models.groot.export.pipeline import GROOT_PIPELINE
from trt.executor.models.groot.inference.pipeline import GROOT_INFERENCE_PIPELINE
from trt.executor.models.groot.load.pipeline import GROOT_LOAD_PIPELINE
from trt.executor.models.pi05.load.pipeline import PI05_LOAD_PIPELINE
from trt.executor.models.smolvla.load.pipeline import SMOLVLA_LOAD_PIPELINE
from trt.executor.benchmark.pipeline import DEFAULT_BENCHMARK

PipelineEntry: TypeAlias = PipelineConfig | Callable[[], PipelineConfig]

_EXPORT: dict[str, PipelineEntry] = {}
_INFERENCE: dict[str, PipelineEntry] = {}
_LOAD: dict[str, LoadPipelineConfig] = {}
_ALIASES: dict[str, str] = {}


def register_export_pipeline(
    model_type: str,
    config: PipelineEntry,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    """Register a TensorRT export pipeline for ``model_type``.

    ``config`` may be a :class:`PipelineConfig` or a zero-arg factory (for lazy
    init). Optional ``aliases`` (e.g. profile ``name`` values) resolve to the
    same canonical key via :func:`get_export_pipeline`.
    """
    key = model_type.strip()
    if key in _EXPORT:
        raise KeyError(f"Export pipeline already registered for {key!r}")
    _EXPORT[key] = config
    for alias in aliases:
        alias_key = alias.strip()
        if alias_key:
            _ALIASES[alias_key] = key


def register_inference_pipeline(
    model_type: str,
    config: PipelineEntry,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    """Register an inference-stage pipeline for ``model_type``.

    Unlike export/load, aliases are also stored directly in ``_INFERENCE`` so
    :func:`get_inference_pipeline` can resolve either the canonical name or an
    alias string.
    """
    key = model_type.strip()
    if key in _INFERENCE:
        raise KeyError(f"Inference pipeline already registered for {key!r}")
    _INFERENCE[key] = config
    for alias in aliases:
        alias_key = alias.strip()
        if alias_key:
            _ALIASES.setdefault(alias_key, key)
            _INFERENCE[alias_key] = config


def register_load_pipeline(
    model_type: str,
    config: LoadPipelineConfig,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    """Register a serialized-engine load pipeline for ``model_type``.

    Used after export to deserialize TRT engines from ``engine_root`` before
    benchmark. Aliases map to the canonical ``model_type`` key only.
    """
    key = model_type.strip()
    if key in _LOAD:
        raise KeyError(f"Load pipeline already registered for {key!r}")
    _LOAD[key] = config
    for alias in aliases:
        alias_key = alias.strip()
        if alias_key:
            _ALIASES[alias_key] = key


def _resolve(entry: PipelineEntry) -> PipelineConfig:
    """Return a concrete config, calling ``entry`` if it is a factory."""
    return entry() if callable(entry) else entry


def _canonical(model_type: str) -> str:
    """Map an alias or canonical name to the registered primary key."""
    return _ALIASES.get(model_type.strip(), model_type.strip())


def get_export_pipeline(model_type: str) -> PipelineConfig:
    """Look up the export pipeline for ``model_type`` (or an alias).

    Raises ``KeyError`` with known types if unregistered.
    """
    key = _canonical(model_type)
    if key not in _EXPORT:
        known = sorted(set(_EXPORT) | set(_ALIASES))
        raise KeyError(f"No export pipeline for {model_type!r}. Known: {known}")
    return _resolve(_EXPORT[key])


def get_inference_pipeline(model_type: str) -> PipelineConfig:
    """Look up the inference pipeline for ``model_type`` (or an alias)."""
    key = _canonical(model_type)
    if key not in _INFERENCE:
        raise KeyError(f"No inference pipeline for {model_type!r}")
    return _resolve(_INFERENCE[key])


def get_benchmark_pipeline(model_type: str | None = None) -> BenchmarkPipelineConfig:
    """Return the benchmark pipeline config.

    Currently model-agnostic; ``model_type`` is ignored and
    ``DEFAULT_BENCHMARK`` is always returned.
    """
    return DEFAULT_BENCHMARK


def get_load_pipeline(model_type: str) -> LoadPipelineConfig:
    """Look up the load pipeline for ``model_type`` (or an alias).

    Raises ``KeyError`` with known types if unregistered.
    """
    key = _canonical(model_type)
    if key not in _LOAD:
        known = sorted(set(_LOAD) | set(_ALIASES))
        raise KeyError(f"No load pipeline for {model_type!r}. Known: {known}")
    return _LOAD[key]


def get_pipeline_for_profile(profile) -> PipelineConfig:
    """Resolve the export pipeline from a profile instance.

    Uses ``profile.pipeline_model_type`` when set, otherwise ``profile.name``.
    """
    model_type = getattr(profile, "pipeline_model_type", None) or getattr(profile, "name", None)
    if not model_type:
        raise ValueError(f"{type(profile).__name__} has no pipeline_model_type or name")
    return get_export_pipeline(model_type)


def _register_builtin() -> None:
    """Register built-in model pipelines at module import time."""
    register_export_pipeline("Gr00tN1d7", GROOT_PIPELINE, aliases=("gr00t", "groot"))
    register_load_pipeline("Gr00tN1d7", GROOT_LOAD_PIPELINE, aliases=("gr00t", "groot"))
    register_inference_pipeline("Gr00tN1d7", GROOT_INFERENCE_PIPELINE, aliases=("gr00t", "groot"))
    register_load_pipeline("pi05", PI05_LOAD_PIPELINE)
    register_load_pipeline("smolvla", SMOLVLA_LOAD_PIPELINE)


_register_builtin()
