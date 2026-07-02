Export Pipeline
===============

The **export pipeline** compiles each stage subgraph to a TensorRT engine under
``engine_root``. It uses :class:`PipelineConfig` with :class:`ExportRunner` per stage.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[ExportPipeline.run]
       PRE[hooks.preprocess]
       LOOP[for each StageConfig]
       RUN[ExportRunner.run]
       ART[ctx.artifacts stage_N]
       POST[hooks.postprocess]

       START --> PRE --> LOOP --> RUN --> ART
       ART --> LOOP
       LOOP --> POST

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,POST nvNode
       class PRE,LOOP,RUN,ART greyNode


Per-stage compile loop
----------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       GLUE[process_inputs]
       PLAN[plan_export<br/>clone + ExportPlan]
       COMPILE[compile<br/>torch.export + TRT]
       SAVE[save_artifacts]
       OUT["engine_root/&lt;subdir&gt;/"]

       GLUE --> PLAN --> COMPILE --> SAVE --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class PLAN,COMPILE nvNode
       class GLUE,SAVE,OUT greyNode


Outputs
-------

Each stage writes a :class:`StageResult` to ``ctx.artifacts["stage_N"]`` containing:

- ``engine_path`` — compiled ``.engine`` file
- ``spec`` — the :class:`ExportPlan` used for compile
- ``tensors`` — representative stage tensors for downstream glue
- ``metadata`` — optional stage metadata

CLI
---

.. code-block:: bash

   python app.py --model gr00t --export-only --engine-dir /tmp/groot_edge_llm

*Files:* ``trt/pipelines/export.py``, ``trt/runner/export.py``, ``trt/compile.py``,
``trt/executor/models/<model>/export/pipeline.py``
