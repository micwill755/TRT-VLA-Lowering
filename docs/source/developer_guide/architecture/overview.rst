Overview
========

**Torch-TRT pipelines** orchestrates VLA export, engine load, inference, and benchmark
through a single :class:`EdgeOrchestrator` and shared :class:`EdgeContext`. Generic
**pipelines** walk ordered **stages**; **runners** execute each stage; model-specific
**hooks** supply the behavior.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       CLIENT[app.py CLI]
       ORCH[Edge Orchestrator]
       PROFILE[VLA Profile]
       CTX[Edge Context]

       subgraph PIPELINES ["Registered pipelines"]
           EXPORT[Export Pipeline]
           LOAD[Load Pipeline]
           BENCH[Benchmark Pipeline]
           INFER[Inference Pipeline]
       end

       CLIENT --> ORCH
       ORCH --> PROFILE
       ORCH --> CTX
       ORCH --> EXPORT
       ORCH --> LOAD
       LOAD --> BENCH
       BENCH --> INFER

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef inputNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef lightSubGraph fill:none,stroke:#aaa,stroke-width:1.5px

       class CLIENT inputNode
       class ORCH,PROFILE,CTX,EXPORT,LOAD,BENCH,INFER nvNode
       class PIPELINES lightSubGraph


Orchestrator flow
-----------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       START[EdgeOrchestrator.run] --> CTX_BUILD[_build_context]
       CTX_BUILD --> MODE{CLI flags}
       MODE -->|default or --export-only| EXPORT[ExportPipeline]
       MODE -->|default or --benchmark-only| LOAD[LoadPipeline]
       LOAD --> BENCH[BenchmarkPipeline]
       BENCH --> INFER[InferencePipeline per backend]

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef inputNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START inputNode
       class CTX_BUILD,EXPORT,LOAD,BENCH,INFER nvNode
       class MODE greyNode


Key files
---------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Area
     - Files
   * - Orchestration
     - ``trt/orchestrator/edge_orchestrator.py``, ``trt/context.py``
   * - Pipeline configs
     - ``trt/config/stage_config.py``, ``trt/config/load_config.py``,
       ``trt/config/benchmark_config.py``, ``trt/config/pipeline_registry.py``
   * - Per-model graphs
     - ``trt/executor/models/<model>/export/pipeline.py``,
       ``trt/executor/models/<model>/inference/pipeline.py``,
       ``trt/executor/models/<model>/load/pipeline.py``
