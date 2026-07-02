Benchmark Pipeline
==================

The **benchmark pipeline** times multiple inference backends on the same prepared
inputs and reports action (and optional logits) parity. It runs after
:class:`LoadPipeline` when benchmarking pre-built engines.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[BenchmarkPipeline.run]
       LOOP[iterations x stages]
       ENABLED{stage.enabled?}
       RUN[stage.run ctx]
       TIME[record timing]
       ACT[record actions]
       REPORT[hooks.report]

       START --> LOOP --> ENABLED
       ENABLED -->|yes| RUN --> TIME --> ACT
       ENABLED -->|no| LOOP
       ACT --> LOOP
       LOOP --> REPORT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,REPORT nvNode
       class LOOP,ENABLED,RUN,TIME,ACT greyNode


Default backends
----------------

The default :class:`BenchmarkPipelineConfig` runs three stages (when enabled):

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       PY[pytorch<br/>ExecutionMode.EAGER]
       MEM[in_memory_trt<br/>ExecutionMode.IN_MEMORY]
       SER[serialized_trt<br/>ExecutionMode.SERIALIZED]

       PY --> MEM --> SER

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       class PY,MEM,SER nvNode


Each backend stage calls :func:`_run_inference`, which sets ``ctx.execution_mode`` and
runs :class:`InferencePipeline` for the registered model type.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage name
     - Enabled when
   * - ``pytorch``
     - Always
   * - ``in_memory_trt``
     - ``ctx.handles.in_memory.vision`` is populated
   * - ``serialized_trt``
     - ``ctx.handles.serialized.vision`` is populated


Reporting
---------

After all iterations, ``hooks.report`` prints timing summaries (warmup excluded) and
parity metrics. GR00T registers ``report_groot_benchmark`` for language logits and
action ADE comparisons.

CLI
---

.. code-block:: bash

   python app.py --model gr00t --benchmark-only --engine-dir /tmp/groot_edge_llm

*Files:* ``trt/pipelines/benchmark.py``, ``trt/config/benchmark_config.py``,
``trt/executor/benchmark/pipeline.py``, ``trt/executor/benchmark/run.py``
