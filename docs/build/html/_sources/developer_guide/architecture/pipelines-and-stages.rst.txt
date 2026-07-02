Pipelines and Stages
====================

What is a pipeline?
-------------------

A **pipeline** is a frozen configuration object that describes an ordered workflow.
Export and inference use :class:`PipelineConfig` (``stages`` + ``hooks``). Load uses
:class:`LoadPipelineConfig`. Benchmark uses :class:`BenchmarkPipelineConfig`.

Each pipeline type has a thin executor class that loops over its stages and delegates
to runners or callables:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Pipeline
     - Executor
     - Config type
   * - Export
     - ``ExportPipeline``
     - ``PipelineConfig``
   * - Inference
     - ``InferencePipeline``
     - ``PipelineConfig``
   * - Load
     - ``LoadPipeline``
     - ``LoadPipelineConfig``
   * - Benchmark
     - ``BenchmarkPipeline``
     - ``BenchmarkPipelineConfig``


What is a stage?
----------------

A **stage** is one node in the pipeline graph. For export and inference, each stage is
a :class:`StageConfig`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``stage_id``
     - Stable integer id (0, 1, 2, …)
   * - ``input_sources``
     - Upstream ``stage_id`` values; ``()`` means entry point
   * - ``runner``
     - Generic executor class (``ExportRunner`` or ``InferenceRunner``)
   * - ``hooks``
     - Model-specific callables the runner invokes
   * - ``engine_subdir``
     - Subdirectory under ``engine_root`` (e.g. ``visual/``, ``language/``)
   * - ``final_output``
     - When true, inference pipeline reads ``actions`` from this stage


Stage graph (GR00T example)
---------------------------

Export and inference share the same stage IDs and ``input_sources`` edges. GR00T uses
four stages:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       S0["Stage 0<br/>visual"]
       S1["Stage 1<br/>language"]
       S2["Stage 2<br/>action_context"]
       S3["Stage 3<br/>action"]

       S0 --> S1
       S1 --> S2
       S2 --> S3

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       class S0,S1,S2,S3 nvNode


PipelineConfig structure
------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       PC[PipelineConfig]
       PH[PipelineHooks<br/>preprocess / postprocess]
       STAGES[tuple of StageConfig]
       S0[StageConfig 0]
       S1[StageConfig 1]
       SN[StageConfig N]

       PC --> PH
       PC --> STAGES
       STAGES --> S0
       STAGES --> S1
       STAGES --> SN

       S0 --> R0[runner class]
       S0 --> H0[StageHooks]
       S0 --> IS0[input_sources]

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class PC,PH,STAGES nvNode
       class S0,S1,SN,R0,H0,IS0 greyNode


Inter-stage data
----------------

Stages communicate through :class:`EdgeContext`:

- **Export** — upstream ``StageResult`` objects in ``ctx.artifacts["stage_N"]``
- **Inference** — upstream ``InferenceStageResult`` in ``ctx.stage_results[N]``

``hooks.process_inputs`` glue functions wire tensors from upstream stages into the
current stage's inputs.

*Files:* ``trt/config/stage_config.py``, ``trt/executor/models/<model>/*/pipeline.py``
