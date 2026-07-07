# trt/config/stage_config.py
"""Pipeline topology for VLA TensorRT export.

Architecture
------------

PipelineConfig (per model, frozen)
    hooks: dict                   whole-pipeline bookends (preprocess / postprocess)
    stages: tuple[StageConfig]  ordered graph nodes

StageConfig (per stage)
    input_sources                 upstream stage ids (graph edges)
    runner                        generic executor class (usually ExportRunner)
    hooks: StageHooks             export-specific logic the runner invokes

Execution (ExportPipeline)
    1. config.hooks["preprocess"](ctx)        once, before the stage loop
    2. for stage in config.stages:
           runner.run(ctx)                     generic TRT compile loop
               hooks.process_inputs(...)      optional inter-stage glue
               hooks.plan_export(...)         clone subgraph, build ExportPlan
               hooks.compile(...)             trace → TRT engine (per stage)
               [trace → compile → artifacts]
               hooks.metadata / save_artifacts / after_stage
           ctx.artifacts[f"stage_{id}"] = result
    3. config.hooks["postprocess"](ctx)       once, after all stages

Unlike vLLM-Omni runtime (load PyTorch model → forward), export requires
hooks.plan_export because subgraphs must be cloned, traced, and compiled to TRT.
Inter-stage glue (hooks.process_inputs) is analogous to omni's custom_process_input_func.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageConfig:
    """One node in the export/inference graph."""

    stage_id: int
    runner: str
    input_sources: tuple[int, ...] = ()
    hooks: dict = field(default_factory=dict)
    stage_name: str | None = None
    result: bool = False
    engine_subdir: str | None = None
    final_output: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    stages: tuple[StageConfig, ...]
    hooks: dict = field(default_factory=dict)
    pipeline_name: str | None = None