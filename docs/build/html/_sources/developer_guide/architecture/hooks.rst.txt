Hooks
=====

What are hooks?
---------------

**Hooks** are model-specific callables registered as dotted import paths on
pipeline and stage configs. Generic runners resolve and invoke them; hooks
contain the actual export and inference logic for each model.

Hooks are resolved at runtime by :func:`resolve` — ``"module.path:callable"`` →
imported function.


Hook resolution
---------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       CFG[StageConfig.hooks]
       PATH["trt.executor.models.groot.export.vision:export"]
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
   * - **Pipeline hooks** — ``preprocess``, ``postprocess``
     - Once before / after the full stage loop. ``preprocess`` returns pipeline
       inputs (tokenized tensors, state, etc.) merged into every stage.
   * - **Stage hooks** — per-stage callables
     - Invoked by the runner for each stage


Stage hook catalog
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Hook
     - Used by
     - Purpose
   * - ``preprocess``
     - Export, Inference
     - Shape stage inputs from merged pipeline + upstream dict
   * - ``export``
     - Export
     - Trace subgraph, compile TRT engine, return ``engine_path`` and
       downstream ``tensors``
   * - ``save_artifacts``
     - Export
     - Write Edge-LLM sidecars (tokenizer, embeddings, etc.)
   * - ``postprocess``
     - Export, Inference
     - Update ``ctx.inference`` / ``ctx.actions`` from stage result
   * - ``compile``
     - Inference
     - On-the-fly TRT compile when ``execution_mode`` is ``IN_MEMORY``
   * - ``load``
     - Inference
     - Deserialize engine wrapper when ``execution_mode`` is ``SERIALIZED``
   * - ``execute``
     - Inference
     - Run eager, in-memory TRT, or serialized TRT forward (mode dispatch
       inside the hook)


Where hooks live
----------------

Model hooks are organized under ``trt/executor/models/<model>/``:

.. code-block:: text

   groot/
     export/
       pipeline.py      # StageConfig + hook paths
       process.py       # pipeline preprocess / postprocess
       vision.py        # preprocess, export, postprocess
       language.py
       action_context.py
       diffusion.py
     inference/
       pipeline.py
       process.py
       vision.py        # preprocess, compile, load, execute, postprocess
       language.py
     load/
       serialize.py     # SerializedGroot* wrappers (not pipeline hooks)

Shared export utilities (not hooks) live in ``trt/modules/export/``,
``trt/compile.py``, and ``trt/plugin/``.

*Config types:* ``trt/config/stage_config.py`` (:class:`PipelineConfig`,
:class:`StageConfig`)
