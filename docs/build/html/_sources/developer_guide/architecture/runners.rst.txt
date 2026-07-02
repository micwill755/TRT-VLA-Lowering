Runners
=======

What is a runner?
-----------------

A **runner** is a generic stage executor. It implements the control flow for one stage
and calls **hooks** for model-specific work. Runners are referenced by dotted import
path on each :class:`StageConfig` (e.g. ``trt.runner.export:ExportRunner``).

Export and inference each have their own runner; load and benchmark do not use the
runner pattern.


ExportRunner
------------

:class:`ExportRunner` drives TensorRT compilation for one export stage:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[ExportRunner.run]
       UP[Read upstream artifacts]
       GLUE[process_inputs hook]
       PLAN[plan_export hook]
       COMPILE[compile hook]
       META[metadata hook]
       SAVE[save_artifacts hook]
       RESULT[StageResult]
       CLEAN[cleanup cloned modules]

       START --> UP --> GLUE --> PLAN --> COMPILE --> META --> SAVE --> RESULT --> CLEAN

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,RESULT nvNode
       class UP,GLUE,PLAN,COMPILE,META,SAVE,CLEAN greyNode


Required export hooks: ``plan_export``, ``compile``. Optional: ``process_inputs``,
``metadata``, ``save_artifacts``, ``after_stage``.

*File:* ``trt/runner/export.py``


InferenceRunner
---------------

:class:`InferenceRunner` executes one inference stage for the active
:class:`ExecutionMode`:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[InferenceRunner.run]
       UP[Read upstream stage_results]
       GLUE[process_inputs hook]
       MODE{execution_mode}
       EAGER[run_eager hook]
       SER[run_serialized hook]
       TRT[run_trt hook]
       STORE[ctx.stage_results id]

       START --> UP --> GLUE --> MODE
       MODE -->|EAGER| EAGER
       MODE -->|SERIALIZED| SER
       MODE -->|IN_MEMORY| TRT
       EAGER --> STORE
       SER --> STORE
       TRT --> STORE

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,STORE nvNode
       class UP,GLUE,MODE,EAGER,SER,TRT greyNode


The runner picks ``run_eager``, ``run_serialized``, or ``run_trt`` based on
``ctx.execution_mode``. Per-stage timing is recorded in ``ctx.inference.stage_ms``.

*File:* ``trt/runner/inference.py``


Runner vs pipeline
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Layer
     - Responsibility
   * - **Pipeline** (``ExportPipeline``, ``InferencePipeline``)
     - Owns the stage loop, calls ``config.hooks.preprocess/postprocess``, stores results
   * - **Runner** (``ExportRunner``, ``InferenceRunner``)
     - Owns one stage's execution loop and dispatches to hooks
   * - **Hooks**
     - Model-specific logic (clone subgraph, compile TRT, run eager/TRT forward)

*Base class:* ``trt/runner/base.py``
