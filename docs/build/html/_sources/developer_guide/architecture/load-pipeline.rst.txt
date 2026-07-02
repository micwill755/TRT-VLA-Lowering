Load Pipeline
=============

The **load pipeline** deserializes compiled TensorRT engines from ``engine_root`` into
``ctx.handles.serialized``. It runs before benchmark when engines were built in a
prior export step (or supplied via ``--engine-dir``).

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[LoadPipeline.run]
       SPECS[SerializedStageSpec list]
       LOAD[load_serialized_modules]
       HANDLES[ctx.handles.serialized]

       START --> SPECS --> LOAD --> HANDLES

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,HANDLES nvNode
       class SPECS,LOAD greyNode


Stage specs
-----------

Each :class:`SerializedStageSpec` maps a handle key to an engine subdirectory and
wrapper class:

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 42

   * - key
     - engine_subdir
     - wrapper
     - Example (GR00T)
   * - ``vision``
     - ``visual/``
     - ``SerializedGrootVision``
     - Vision TRT engine
   * - ``language``
     - ``language/``
     - ``SerializedGrootLanguage``
     - Language TRT engine
   * - ``action_context``
     - ``action_context/``
     - ``SerializedGrootActionContext``
     - Context splice engine
   * - ``action``
     - ``action/``
     - ``SerializedGrootAction``
     - Action / diffusion engine


Directory layout
----------------

.. code-block:: text

   engine_root/
     visual/engine.trt + config.json
     language/engine.trt + config.json
     action_context/engine.trt + config.json
     action/engine.trt + config.json

Load does **not** use runners or stage hooks — it is a straight deserialize step.
Inference ``run_serialized`` hooks call into the loaded wrappers.

*Files:* ``trt/pipelines/load.py``, ``trt/config/load_config.py``,
``trt/serialize.py``, ``trt/executor/models/<model>/load/pipeline.py``
