# trt/config/stage_config.py
"""Pipeline topology for VLA TensorRT export.

Architecture
------------

PipelineConfig (per model, frozen)
    hooks: PipelineHooks          whole-pipeline bookends (preprocess / postprocess)
    stages: tuple[StageConfig]  ordered graph nodes

StageConfig (per stage)
    input_sources                 upstream stage ids (graph edges)
    kind                          VISION_ENCODE | LANGUAGE_PREFILL | ACTION_ROLLOUT | ...
    runner                        generic executor class (usually ExportRunner)
    hooks: StageHooks             export-specific logic the runner invokes
    io                            TRT engine I/O names/shapes

Execution (ExportPipeline)
    1. config.hooks.preprocess(ctx)           once, before the stage loop
    2. for stage in config.stages:
           runner.run(ctx)                     generic TRT compile loop
               hooks.process_inputs(...)      optional inter-stage glue
               hooks.plan_export(...)         clone subgraph, build ExportPlan
               hooks.compile(...)             trace → TRT engine (per stage)
               [trace → compile → artifacts]
               hooks.metadata / save_artifacts / after_stage
           ctx.artifacts[f"stage_{id}"] = result
    3. config.hooks.postprocess(ctx)          once, after all stages

Unlike vLLM-Omni runtime (load PyTorch model → forward), export requires
hooks.plan_export because subgraphs must be cloned, traced, and compiled to TRT.
Inter-stage glue (hooks.process_inputs) is analogous to omni's custom_process_input_func.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trt.io_spec import ComponentIOSpec, PipelineIOSpec


class StageKind(str, Enum):
    VISION_ENCODE = "vision_encode"
    LANGUAGE_PREFILL = "language_prefill"
    ACTION_ROLLOUT = "action_rollout"
    ACTION_CONTEXT = "action_context"


@dataclass(frozen=True)
class PipelineHooks:
    """Whole-pipeline hooks for one model (defined in executor/models/<model>/)."""

    preprocess: str | None = None   # normalize ctx.model_inputs before stage 0
    postprocess: str | None = None  # parity / smoke after all engines are built


@dataclass(frozen=True)
class StageHooks:
    """Per-stage hooks; export uses compile hooks, inference uses run_* hooks."""

    plan_export: str | None = None
    compile: str | None = None
    run_eager: str | None = None
    run_serialized: str | None = None
    run_trt: str | None = None
    process_inputs: str | None = None
    save_artifacts: str | None = None
    metadata: str | None = None
    after_stage: str | None = None


@dataclass(frozen=True)
class StageConfig:
    """One node in the export graph."""

    stage_id: int
    kind: StageKind
    input_sources: tuple[int, ...]  # upstream stage ids; () = entry point
    runner: str                     # e.g. "trt.runner.export:ExportRunner"
    hooks: StageHooks
    io: ComponentIOSpec
    final_output: bool = False
    engine_subdir: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen stage graph for one VLA (export, inference, or both)."""

    model_type: str
    stages: tuple[StageConfig, ...]
    hooks: PipelineHooks = field(default_factory=PipelineHooks)
    io: PipelineIOSpec | None = None