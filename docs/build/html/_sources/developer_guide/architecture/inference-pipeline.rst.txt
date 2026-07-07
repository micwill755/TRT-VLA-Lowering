Inference Pipeline
==================

The **inference pipeline** runs the model stage graph end-to-end for one sample.
It uses the same :class:`PipelineConfig` topology and top-level loop as export,
but with :class:`InferenceRunner` and mode-aware stage hooks.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[InferencePipeline.run]
       PRE[pipeline preprocess]
       MERGE[merge pipeline_inputs + upstream]
       RUN[InferenceRunner.run]
       OUT[stage_outputs stage_id]
       POST[pipeline postprocess]

       START --> PRE --> MERGE --> RUN --> OUT
       OUT --> MERGE
       MERGE --> POST

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,POST nvNode
       class PRE,MERGE,RUN,OUT greyNode


Per-stage inference loop
------------------------

:class:`InferenceRunner` dispatches compile/load work by execution mode, then
calls a single ``execute`` hook:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[InferenceRunner.run]
       PRE[preprocess]
       MODE{execution_mode}
       COMP[compile hook]
       LOAD[load hook]
       EXEC[execute hook]
       POST[postprocess]

       START --> PRE --> MODE
       MODE -->|IN_MEMORY| COMP --> EXEC
       MODE -->|SERIALIZED| LOAD --> EXEC
       MODE -->|EAGER| EXEC
       EXEC --> POST

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,EXEC nvNode
       class PRE,MODE,COMP,LOAD,POST greyNode


Execution modes
---------------

Each stage's ``execute`` hook branches on ``ctx.execution_mode``:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Mode
     - Runner hook (before execute)
     - Backend
   * - ``EAGER``
     - none
     - Live PyTorch ``ctx.model`` / ``ctx.policy`` (language uses stock HF
       ``language_model`` with model weight dtype)
   * - ``IN_MEMORY``
     - ``compile``
     - On-the-fly Torch-TensorRT module compiled in-process
   * - ``SERIALIZED``
     - ``load``
     - Deserialized engine wrapper from ``ctx.engine_root/<subdir>/``


GR00T stage sequence
--------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       PRE[preprocess]
       V[vision]
       L[language]
       AC[action_context]
       A[action]

       PRE --> V --> L --> AC --> A

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       class PRE,V,L,AC,A nvNode


Scratch state lives in ``ctx.inference`` (image embeddings, language outputs,
noise, per-stage timings). Stage dicts accumulate in ``ctx.stage_results``.
The final ``action`` stage writes ``actions`` to ``ctx.actions`` via its
``postprocess`` hook.

Invocation
----------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Entry point
     - When
   * - ``--inference-only``
     - CLI runs :class:`InferencePipeline` once in ``EAGER`` mode
   * - :class:`BenchmarkPipeline`
     - Runs inference once per backend (eager, in-memory TRT, serialized TRT)

On completion the pipeline prints ``Pipeline complete in X.XXs``.

*Files:* ``trt/pipelines/inference.py``, ``trt/runner/inference.py``,
``trt/executor/models/<model>/inference/pipeline.py``
