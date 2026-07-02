Hooks
=====

What are hooks?
---------------

**Hooks** are model-specific callables registered as dotted import paths on pipeline
and stage configs. Generic runners resolve and invoke them; hooks contain the actual
export, compile, and inference logic for each model.

Hooks are resolved at runtime by :func:`resolve` — ``"module.path:callable"`` → imported
function.


Hook resolution
---------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       CFG[StageConfig.hooks]
       PATH["trt.executor.models.groot.export.vision:plan_export"]
       RESOLVE[resolve path]
       FN[callable]
       RUNNER[ExportRunner / InferenceRunner]

       CFG --> PATH --> RESOLVE --> FN
       RUNNER -->|calls| FN

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class CFG,RESOLVE,RUNNER nvNode
       class PATH,FN greyNode


*File:* ``trt/hooks/resolve.py``


Pipeline hooks vs stage hooks
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - When invoked
   * - **PipelineHooks** — ``preprocess``, ``postprocess``
     - Once before / after the full stage loop
   * - **StageHooks** — per-stage callables
     - Invoked by the runner for each stage


Stage hook catalog
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Hook
     - Used by
     - Purpose
   * - ``process_inputs``
     - Export, Inference
     - Glue upstream tensors into current stage inputs
   * - ``plan_export``
     - Export
     - Clone subgraph, build :class:`ExportPlan`
   * - ``compile``
     - Export
     - ``torch.export`` trace + Torch-TensorRT compile
   * - ``save_artifacts``
     - Export
     - Write Edge-LLM sidecars (tokenizer, embeddings, etc.)
   * - ``metadata``
     - Export
     - Attach stage metadata to ``StageResult``
   * - ``run_eager``
     - Inference
     - PyTorch forward on live ``ctx.model`` / ``ctx.policy``
   * - ``run_serialized``
     - Inference
     - Forward through ``ctx.handles.serialized``
   * - ``run_trt``
     - Inference
     - Forward through in-memory TRT module
   * - ``after_stage``
     - Export, Inference
     - Optional post-stage side effects


Where hooks live
----------------

Model hooks are organized under ``trt/executor/models/<model>/``:

.. code-block:: text

   groot/
     export/
       pipeline.py      # StageConfig + hook paths
       vision.py        # plan_export, compile
       language.py
       glue.py          # process_inputs
     inference/
       pipeline.py
       vision.py        # run_eager, run_serialized, run_trt
       glue.py

Shared export utilities (not hooks) live in ``trt/modules/export/``, ``trt/compile.py``,
and ``trt/plugin/``.

*Config types:* ``trt/config/stage_config.py`` (:class:`PipelineHooks`, :class:`StageHooks`)
