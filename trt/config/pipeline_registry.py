"""Pipeline registry: model_type → PipelineConfig.

Usage::

    from trt.config.pipeline_registry import get_pipeline, register_pipeline

    config = get_pipeline("Gr00tN1d7")
    VLAExportPipeline(config).run(profile, policy, device, model_inputs, engine_root)

To add a new VLA:
    1. Define ``MY_MODEL_PIPELINE`` in ``trt/executor/models/<model>/pipeline.py``.
    2. Register it below (or call ``register_pipeline`` from that module).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from trt.config.stage_config import PipelineConfig

from trt.executor.models.gr00t.pipeline import GROOT_PIPELINE
from trt.executor.models.molmo2.molmo2 import MOLMO2_PIPELINE

PipelineEntry: TypeAlias = PipelineConfig | Callable[[], PipelineConfig]

_REGISTRY: dict[str, PipelineEntry] = {}
_ALIASES: dict[str, str] = {}

def register_pipeline(
    model_type: str,
    config: PipelineEntry,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    """Register a pipeline config under ``model_type`` and optional aliases."""
    key = model_type.strip()
    if key in _REGISTRY:
        raise KeyError(f"Pipeline already registered for {key!r}")

    _REGISTRY[key] = config

    for alias in aliases:
        alias_key = alias.strip()
        if not alias_key:
            continue
        if alias_key in _ALIASES and _ALIASES[alias_key] != key:
            raise KeyError(f"Alias {alias_key!r} already maps to {_ALIASES[alias_key]!r}")
        _ALIASES[alias_key] = key

def resolve_pipeline(entry: PipelineEntry) -> PipelineConfig:
    if callable(entry):
        return entry()
    return entry

def get_pipeline(model_type: str) -> PipelineConfig:
    """Look up a registered pipeline by ``model_type`` or alias."""
    key = model_type.strip()
    canonical = _ALIASES.get(key, key)
    if canonical not in _REGISTRY:
        known = sorted(set(_REGISTRY) | set(_ALIASES))
        raise KeyError(f"No pipeline registered for {model_type!r}. Known: {known}")
    return resolve_pipeline(_REGISTRY[canonical])

def get_pipeline_for_profile(profile) -> PipelineConfig:
    """Resolve pipeline from ``profile.pipeline_model_type`` or ``profile.name``."""
    model_type = getattr(profile, "pipeline_model_type", None) or getattr(profile, "name", None)
    if not model_type:
        raise ValueError(f"{type(profile).__name__} has no pipeline_model_type or name")
    return get_pipeline(model_type)

def list_pipelines() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))

def _register_builtin_pipelines() -> None:
    register_pipeline("Gr00tN1d7", GROOT_PIPELINE, aliases=("gr00t", "groot"))
    register_pipeline("MolmoAct2", MOLMO2_PIPELINE, aliases=("molmo2", "molmoact2"))

_register_builtin_pipelines()