Pipeline Runtime
================

Architecture
------------

**Torch-TRT pipelines** orchestrates VLA export, engine load, and benchmark/inference
through a single :class:`EdgeOrchestrator` and shared :class:`EdgeContext`. Model-specific
behavior lives in **profiles** and per-stage **hooks**; generic stage runners execute
export, load, and inference graphs registered per model type.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       CLIENT[app.py CLI]
       ORCH[Edge Orchestrator]
       PROFILE[VLA Profile]
       CTX[Edge Context]
       EXPORT[Export Pipeline]
       LOAD[Load Pipeline]
       BENCH[Benchmark Pipeline]
       INFER[Inference Pipeline]
       HOOKS[Stage Hooks]
       ENGINES[TRT Engines]

       CLIENT -->|run| ORCH
       ORCH -->|builds| PROFILE
       ORCH -->|owns| CTX
       ORCH -->|export-only or full| EXPORT
       ORCH -->|benchmark path| LOAD
       LOAD --> BENCH
       BENCH --> INFER
       EXPORT -->|compile| ENGINES
       LOAD -->|deserialize| ENGINES
       EXPORT --> HOOKS
       INFER --> HOOKS
       HOOKS -->|execute| ENGINES

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef inputNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class CLIENT inputNode
       class ORCH,PROFILE,CTX,EXPORT,LOAD,BENCH,INFER,HOOKS nvNode
       class ENGINES greyNode


Key Components
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Component
     - Description
   * - **Edge Orchestrator**
     - CLI entry driver. Builds :class:`EdgeContext`, selects export and/or benchmark paths
       from ``--export-only`` / ``--benchmark-only``. *Files:* ``trt/orchestrator/edge_orchestrator.py``
   * - **VLA Profile**
     - Per-model policy: loads weights/tokenizers, prepares compile inputs, names engine
       directories. *Files:* ``trt/profile.py``, ``trt/executor/models/<model>/profile.py``
   * - **Edge Context**
     - Shared session state passed through all pipelines: ``artifacts``, ``handles``,
       ``inference``, ``benchmark``. *Files:* ``trt/context.py``
   * - **Export Pipeline**
     - Stage graph for ONNX trace + TensorRT compile. Each stage produces a
       ``StageResult`` under ``engine_root``. *Files:* ``trt/pipelines/export.py``,
       ``trt/executor/export/runner.py``
   * - **Load Pipeline**
     - Deserializes compiled engines into ``ctx.handles.serialized`` for benchmark and
       parity runs. *Files:* ``trt/pipelines/load.py``
   * - **Benchmark Pipeline**
     - Runs eager, serialized TRT, and optional in-memory TRT backends; records timings
       and action parity. *Files:* ``trt/pipelines/benchmark.py``,
       ``trt/executor/benchmark/run.py``
   * - **Inference Pipeline**
     - Executes the per-model inference stage graph (preprocess → vision → language →
       action). *Files:* ``trt/pipelines/inference.py``,
       ``trt/executor/inference/runner.py``
   * - **Stage Hooks**
     - Model-specific ``plan_export``, ``compile``, ``run_eager``, ``run_serialized``,
       ``run_trt`` implementations. *Files:* ``trt/executor/models/<model>/``
   * - **TRT Engines**
     - Edge-LLM–compatible engines (e.g. GR00T: ``visual/``, ``llm/``, ``action_context/``,
       ``action/``). Built at export; consumed at load/inference.


Inference Workflow
------------------

End-to-end VLA inference is a **linear stage graph**. Each stage selects an execution
backend via :class:`ExecutionMode` (eager PyTorch, serialized TRT, or in-memory TRT).

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       INPUT[Dataset Sample] --> PRE[Preprocess]
       PRE --> VISION[Vision Stage]
       VISION --> LANG[Language Stage]
       LANG --> CTX_STAGE[Action Context]
       CTX_STAGE --> ACTION[Action Stage]
       ACTION --> OUT[Policy Actions]

       subgraph BACKENDS ["Per-stage backend"]
           EAGER[Eager PyTorch]
           SER[Serialized TRT]
           MEM[In-memory TRT]
       end

       VISION -.-> BACKENDS
       LANG -.-> BACKENDS
       ACTION -.-> BACKENDS

       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef nvLightNode fill:#b8d67e,stroke:#76B900,stroke-width:1px,color:#333
       classDef inputNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef lightSubGraph fill:none,stroke:#aaa,stroke-width:1.5px

       class INPUT inputNode
       class PRE,VISION,LANG,CTX_STAGE,ACTION greyNode
       class OUT nvLightNode
       class EAGER,SER,MEM nvNode
       class BACKENDS lightSubGraph


Export Workflow
---------------

Export walks the same stage IDs in order. For each stage the runner calls
``plan_export`` → ``compile`` (TensorRT + Edge-LLM plugins) → ``save_artifacts``.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       SAMPLE[prepare_compile_inputs] --> PRE_E[Preprocess]
       PRE_E --> S0[Vision Export]
       S0 --> S1[Language Export]
       S1 --> S2[Action Context Export]
       S2 --> S3[Action Export]
       S3 --> ROOT[engine_root/]

       subgraph STAGE ["Per export stage"]
           PLAN[plan_export]
           COMPILE[compile TRT]
           SAVE[save_artifacts]
           PLAN --> COMPILE --> SAVE
       end

       S0 -.-> STAGE
       S1 -.-> STAGE
       S2 -.-> STAGE
       S3 -.-> STAGE

       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef nvLightNode fill:#b8d67e,stroke:#76B900,stroke-width:1px,color:#333
       classDef inputNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef lightSubGraph fill:none,stroke:#aaa,stroke-width:1.5px

       class SAMPLE inputNode
       class PRE_E,S0,S1,S2,S3 greyNode
       class ROOT nvLightNode
       class PLAN,COMPILE,SAVE nvNode
       class STAGE lightSubGraph
