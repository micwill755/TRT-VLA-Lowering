from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from trt.config.benchmark_config import BenchmarkPipelineConfig
from trt.config.inference_config import HandleSource, InferenceRunConfig, InferenceRunHooks
from trt.config.load_config import LoadPipelineConfig
from trt.config.stage_config import PipelineConfig
from trt.executor.models.groot.export.pipeline import GROOT_PIPELINE
from trt.executor.models.groot.inference.pipeline import GROOT_INFERENCE_PIPELINE
from trt.executor.models.groot.inference.run import run_eager, run_trt
from trt.executor.models.groot.load.pipeline import GROOT_LOAD_PIPELINE
from trt.pipelines.benchmark import default_groot_benchmark_config

PipelineEntry: TypeAlias = PipelineConfig | Callable[[], PipelineConfig]

_EXPORT: dict[str, PipelineEntry] = {}
_INFERENCE: dict[str, PipelineEntry] = {}
_BENCHMARK: dict[str, BenchmarkPipelineConfig] = {}
_LOAD: dict[str, LoadPipelineConfig] = {}
_ALIASES: dict[str, str] = {}


def register_export_pipeline(
    model_type: str,
    config: PipelineEntry,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
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
    key = model_type.strip()
    if key in _INFERENCE:
        raise KeyError(f"Inference pipeline already registered for {key!r}")
    _INFERENCE[key] = config
    for alias in aliases:
        alias_key = alias.strip()
        if alias_key:
            _ALIASES.setdefault(alias_key, key)
            _INFERENCE[alias_key] = config


def register_benchmark_pipeline(model_type: str, config: BenchmarkPipelineConfig) -> None:
    _BENCHMARK[model_type] = config


def register_load_pipeline(
    model_type: str,
    config: LoadPipelineConfig,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    key = model_type.strip()
    if key in _LOAD:
        raise KeyError(f"Load pipeline already registered for {key!r}")
    _LOAD[key] = config
    for alias in aliases:
        alias_key = alias.strip()
        if alias_key:
            _ALIASES[alias_key] = key


def _resolve(entry: PipelineEntry) -> PipelineConfig:
    return entry() if callable(entry) else entry


def _canonical(model_type: str) -> str:
    return _ALIASES.get(model_type.strip(), model_type.strip())


def get_export_pipeline(model_type: str) -> PipelineConfig:
    key = _canonical(model_type)
    if key not in _EXPORT:
        known = sorted(set(_EXPORT) | set(_ALIASES))
        raise KeyError(f"No export pipeline for {model_type!r}. Known: {known}")
    return _resolve(_EXPORT[key])


def get_inference_pipeline_config(model_type: str) -> PipelineConfig:
    key = _canonical(model_type)
    if key not in _INFERENCE:
        raise KeyError(f"No inference pipeline for {model_type!r}")
    return _resolve(_INFERENCE[key])


def get_inference_pipeline(model_type: str, handle_source: HandleSource) -> InferenceRunConfig:
    key = _canonical(model_type)
    if key not in _INFERENCE:
        raise KeyError(f"No inference pipeline for {model_type!r}")
    return InferenceRunConfig(
        handle_source=handle_source,
        hooks=InferenceRunHooks(run=run_trt),
    )


def get_eager_runner(model_type: str):
    key = _canonical(model_type)
    if key not in _INFERENCE:
        raise KeyError(f"No eager runner for {model_type!r}")
    return run_eager


def get_benchmark_pipeline(model_type: str) -> BenchmarkPipelineConfig:
    key = _canonical(model_type)
    if key in _BENCHMARK:
        return _BENCHMARK[key]
    return default_groot_benchmark_config()


def get_load_pipeline(model_type: str) -> LoadPipelineConfig:
    key = _canonical(model_type)
    if key not in _LOAD:
        known = sorted(set(_LOAD) | set(_ALIASES))
        raise KeyError(f"No load pipeline for {model_type!r}. Known: {known}")
    return _LOAD[key]


def get_pipeline_for_profile(profile) -> PipelineConfig:
    model_type = getattr(profile, "pipeline_model_type", None) or getattr(profile, "name", None)
    if not model_type:
        raise ValueError(f"{type(profile).__name__} has no pipeline_model_type or name")
    return get_export_pipeline(model_type)


def _register_builtin() -> None:
    from trt.config.load_config import SerializedStageSpec
    from trt.export.pi05 import PI05VisionEngineAdapter
    from trt.export.smolvla import SerializedSmolVLAVision
    from trt.export.molmoact2 import SerializedMolmoAct2Action, SerializedMolmoAct2Backbone
    from trt.serialize import (
        SerializedGrootAction,
        SerializedGrootActionContext,
        SerializedGrootLanguage,
        SerializedGrootVision,
        SerializedPI05Action,
        SerializedPI05Language,
    )

    register_export_pipeline("Gr00tN1d7", GROOT_PIPELINE, aliases=("gr00t", "groot"))
    register_load_pipeline("Gr00tN1d7", GROOT_LOAD_PIPELINE, aliases=("gr00t", "groot"))
    register_inference_pipeline("Gr00tN1d7", GROOT_INFERENCE_PIPELINE, aliases=("gr00t", "groot"))
    register_benchmark_pipeline("Gr00tN1d7", default_groot_benchmark_config())

    register_load_pipeline(
        "pi05",
        LoadPipelineConfig(
            stages=(
                SerializedStageSpec("vision", "visual", PI05VisionEngineAdapter),
                SerializedStageSpec("language", "language", SerializedPI05Language),
                SerializedStageSpec("action", "action", SerializedPI05Action),
            ),
        ),
    )
    register_load_pipeline(
        "smolvla",
        LoadPipelineConfig(
            stages=(
                SerializedStageSpec("vision", "visual", SerializedSmolVLAVision),
                SerializedStageSpec("language", "language", SerializedPI05Language),
                SerializedStageSpec("action", "action", SerializedPI05Action),
            ),
        ),
    )
    register_load_pipeline(
        "molmoact2",
        LoadPipelineConfig(
            stages=(
                SerializedStageSpec("language", "language", SerializedMolmoAct2Backbone),
                SerializedStageSpec("action", "action", SerializedMolmoAct2Action),
            ),
        ),
    )


_register_builtin()
