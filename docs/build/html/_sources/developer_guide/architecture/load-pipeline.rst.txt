Engine Load (Serialized Inference)
==================================

Serialized TensorRT engines are **not** loaded by a separate pipeline step.
Instead, each inference stage's ``load`` hook constructs a wrapper from
``ctx.engine_root`` when ``ctx.execution_mode`` is ``SERIALIZED``.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       MODE[ExecutionMode.SERIALIZED]
       LOAD[stage load hook]
       WRAP[SerializedGroot* wrapper]
       EXEC[stage execute hook]

       MODE --> LOAD --> WRAP --> EXEC

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class MODE,EXEC nvNode
       class LOAD,WRAP greyNode


GR00T wrappers
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 42

   * - Stage
     - engine_subdir
     - Wrapper class
     - Engine file
   * - vision
     - ``visual/``
     - ``SerializedGrootVision``
     - ``visual.engine``
   * - language
     - ``language/``
     - ``SerializedGrootLanguage``
     - ``language.engine``
   * - action_context
     - ``action_context/``
     - ``SerializedGrootActionContext``
     - ``context.engine``
   * - action
     - ``action/``
     - ``SerializedGrootAction``
     - ``action.engine``


Directory layout
----------------

.. code-block:: text

   engine_root/
     visual/visual.engine + config.json
     language/language.engine + config.json + tokenizer + embedding.safetensors
     action_context/context.engine + config.json
     action/action.engine + config.json

Each ``load`` hook reads ``config.json`` via :class:`SerializedTRTEngine` and
returns a model-specific callable used by ``execute``.

The benchmark pipeline checks that all four engine directories exist before
running the ``SERIALIZED`` mode.

*Files:* ``trt/serialize.py``, ``trt/executor/models/<model>/load/serialize.py``,
``trt/executor/models/<model>/inference/*.py`` (``load`` hooks)
