Export Pipeline
===============

The **export pipeline** compiles each stage subgraph to a TensorRT engine under
``engine_root``. It uses :class:`PipelineConfig` with :class:`ExportRunner` per
stage. Export and inference share the same top-level loop: pipeline-level
``preprocess`` → stage loop with upstream merge → pipeline-level ``postprocess``.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[ExportPipeline.run]
       PRE[pipeline preprocess]
       MERGE[merge pipeline_inputs + upstream]
       RUN[ExportRunner.run]
       OUT[stage_outputs stage_id]
       POST[pipeline postprocess]

       START --> PRE --> MERGE --> RUN --> OUT
       OUT --> MERGE
       MERGE --> POST

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,POST nvNode
       class PRE,MERGE,RUN,OUT greyNode


Per-stage export loop
---------------------

Each :class:`ExportRunner` runs the same hook sequence for every stage:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       PRE[preprocess]
       EXP[export<br/>trace + TRT compile]
       SAVE[save_artifacts?]
       POST[postprocess]
       OUT["engine_root/&lt;subdir&gt;/"]

       PRE --> EXP --> SAVE --> POST --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class EXP nvNode
       class PRE,SAVE,POST,OUT greyNode


Stage inputs are built by the pipeline executor, not by a separate glue hook:

1. Start from pipeline-level inputs (``ctx.model_inputs`` normalized by pipeline
   ``preprocess``).
2. Merge the upstream stage dict when ``input_sources`` is set.
3. Pass the merged dict to :class:`ExportRunner`.

Each stage's ``preprocess`` hook shapes tensors for that engine boundary.
Cross-stage wiring (for example splicing ``image_embs`` into language
``inputs_embeds``) lives in the downstream stage's ``preprocess``, not in a
generic runner.


Outputs
-------

Each stage returns a dict stored at ``ctx.stage_results[stage_id]``. Typical keys:

- ``engine_path`` — compiled ``.engine`` file under ``engine_root/<subdir>/``
- ``tensors`` — representative stage outputs passed to downstream stages
  (for example ``image_embs``, ``lm_hidden``, ``context_embs``, ``actions``)
- ``metadata`` — optional stage metadata

The language stage additionally uses ``save_artifacts`` to write Edge-LLM
sidecars (tokenizer JSON, embedding table) beside ``language.engine``.

Export stages use dummy or eager-forward tensors where needed so the pipeline
does not re-run compiled modules after export.


CLI
---

.. code-block:: bash

   python app.py --model gr00t --export-only --engine-dir /tmp/groot_edge_llm

On completion the pipeline prints ``Pipeline complete in X.XXs``.

*Files:* ``trt/pipelines/export.py``, ``trt/runner/export.py``, ``trt/compile.py``,
``trt/executor/models/<model>/export/pipeline.py``
