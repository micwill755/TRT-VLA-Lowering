Inference Pipeline
==================

The **inference pipeline** runs the model stage graph end-to-end for one sample.
It uses the same :class:`StageConfig` topology as export, but with
:class:`InferenceRunner` and ``run_*`` hooks instead of ``plan_export`` / ``compile``.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[InferencePipeline.run]
       PRE[hooks.preprocess]
       LOOP[for each StageConfig]
       RUN[InferenceRunner.run]
       FINAL[final_output stage]
       POST[hooks.postprocess]
       ACT[ctx.actions]

       START --> PRE --> LOOP --> RUN
       RUN --> LOOP
       LOOP --> FINAL --> ACT
       FINAL --> POST

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,ACT nvNode
       class PRE,LOOP,RUN,FINAL,POST greyNode


Execution modes
---------------

Each stage's :class:`InferenceRunner` dispatches to one hook based on
``ctx.execution_mode``:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Mode
     - Hook
     - Backend
   * - ``EAGER``
     - ``run_eager``
     - Live PyTorch ``ctx.model`` / ``ctx.policy``
   * - ``SERIALIZED``
     - ``run_serialized``
     - Deserialized engines in ``ctx.handles.serialized``
   * - ``IN_MEMORY``
     - ``run_trt``
     - In-process TRT modules in ``ctx.handles.in_memory``


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


Scratch state lives in ``ctx.inference`` (tokenized inputs, image embeddings, language
outputs, noise, per-stage timings). Results accumulate in ``ctx.stage_results``.

Invocation
----------

Inference is not called directly from the CLI. The **benchmark pipeline** invokes it
once per backend (eager, serialized TRT, in-memory TRT).

*Files:* ``trt/pipelines/inference.py``, ``trt/runner/inference.py``,
``trt/executor/models/<model>/inference/pipeline.py``
