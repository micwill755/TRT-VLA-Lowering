Runners
=======

What is a runner?
-----------------

A **runner** is a generic stage executor. It implements the control flow for one
stage and calls **hooks** for model-specific work. Runners are referenced by
dotted import path on each :class:`StageConfig` (e.g.
``trt.runner.export:ExportRunner``).

Export and inference each have their own runner; benchmark does not use the
runner pattern.


ExportRunner
------------

:class:`ExportRunner` drives TensorRT compilation for one export stage:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[ExportRunner.run]
       PRE[preprocess hook]
       EXP[export hook]
       SAVE[save_artifacts hook?]
       POST[postprocess hook]
       RESULT[stage dict]

       START --> PRE --> EXP --> SAVE --> POST --> RESULT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,RESULT nvNode
       class PRE,EXP,SAVE,POST greyNode


Required export hooks: ``export``. Optional: ``preprocess``, ``save_artifacts``,
``postprocess``.

The ``export`` hook owns subgraph selection, attention patching, TRT trace/compile,
and writing ``config.json`` sidecars. It returns ``engine_path``, ``tensors``,
and optional ``metadata`` merged into the stage result.

*File:* ``trt/runner/export.py``


InferenceRunner
---------------

:class:`InferenceRunner` executes one inference stage for the active
:class:`ExecutionMode`:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[InferenceRunner.run]
       PRE[preprocess hook]
       MODE{execution_mode}
       COMP[compile hook]
       LOAD[load hook]
       EXEC[execute hook]
       POST[postprocess hook]
       RESULT[stage dict]

       START --> PRE --> MODE
       MODE -->|IN_MEMORY| COMP --> EXEC
       MODE -->|SERIALIZED| LOAD --> EXEC
       MODE -->|EAGER| EXEC
       EXEC --> POST --> RESULT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,RESULT nvNode
       class PRE,MODE,COMP,LOAD,EXEC,POST greyNode


Required inference hooks: ``execute``. Optional: ``preprocess``, ``compile``,
``load``, ``postprocess``.

The ``execute`` hook dispatches internally to eager, in-memory TRT, or serialized
TRT backends. ``compile`` runs only for ``IN_MEMORY``; ``load`` runs only for
``SERIALIZED`` and typically constructs a wrapper from
``SerializedTRTEngine(ctx.engine_root / <subdir>)``.

*File:* ``trt/runner/inference.py``


Runner vs pipeline
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Layer
     - Responsibility
   * - **Pipeline** (``ExportPipeline``, ``InferencePipeline``)
     - Owns the stage loop, merges upstream outputs, calls pipeline
       ``preprocess`` / ``postprocess``, stores ``ctx.stage_results``
   * - **Runner** (``ExportRunner``, ``InferenceRunner``)
     - Owns one stage's hook sequence
   * - **Hooks**
     - Model-specific logic (prepare tensors, compile TRT, run forward)

*Shared types:* ``trt/config/stage_config.py``
