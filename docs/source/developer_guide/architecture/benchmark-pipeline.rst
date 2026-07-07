Benchmark Pipeline
==================

The **benchmark pipeline** runs the inference pipeline once per execution mode on
the same prepared inputs and reports per-stage tensor parity against eager
PyTorch. It does not run a separate load step — serialized engines are loaded
per stage by each stage's ``load`` hook when ``ctx.execution_mode`` is
``SERIALIZED``.

Flow
----

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       START[BenchmarkPipeline.run]
       LOOP[for each ExecutionMode]
       INF[InferencePipeline.run]
       SNAP[record_stage tensors]
       PARITY[report_stage_parity]

       START --> LOOP --> INF --> SNAP --> LOOP
       LOOP --> PARITY

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class START,PARITY nvNode
       class LOOP,INF,SNAP greyNode


Backends
--------

The benchmark runs three modes in order (serialized is skipped when engine
directories are missing):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Mode
     - Description
   * - ``eager``
     - Reference PyTorch path (``ExecutionMode.EAGER``)
   * - ``in_memory``
     - On-the-fly TRT compile per stage (``ExecutionMode.IN_MEMORY``)
   * - ``serialized``
     - Engines from ``--engine-dir`` (``ExecutionMode.SERIALIZED``)


Each run sets a fixed seed (``SEED = 42``), clears ``ctx.stage_results``, and
snapshots each stage's output tensors into ``ctx.benchmark.stage_tensors``.


Reporting
---------

After all backends complete, ``report_stage_parity`` prints aligned tables
comparing eager against each TRT backend. GR00T registers these parity tensors
in ``STAGE_PARITY_TENSORS``:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Stage
     - Tensor key
     - Metric columns
   * - ``vision``
     - ``image_embs``
     - mean abs, max abs, rel L2, rel mean %, close %
   * - ``language``
     - ``lm_hidden``
     - same
   * - ``action_context``
     - ``context_embs``
     - same
   * - ``action``
     - ``actions``
     - same

Example output:

.. code-block:: text

   Parity: eager vs in_memory
   --------------------------------------------------------------------------------
   Stage            Tensor           Mean Abs    Max Abs   Rel L2  Rel Mean   Close
   --------------------------------------------------------------------------------
   vision           image_embs       0.006502   0.358398   0.0206     1.15%   90.0%
   language         lm_hidden        0.090187  12.605469   0.0770     7.45%   21.1%
   ...

``--warmup`` and ``--num-iterations`` are reserved on the CLI for future timing
loops; the current benchmark pipeline runs one inference pass per mode for
parity only.


CLI
---

.. code-block:: bash

   # Export + benchmark (default)
   python app.py --model gr00t --engine-dir /tmp/groot_edge_llm

   # Benchmark only (engines already exported)
   python app.py --model gr00t --benchmark-only --engine-dir /tmp/groot_edge_llm

*Files:* ``trt/pipelines/benchmark.py``, ``trt/measure.py``,
``trt/executor/models/groot/inference/pipeline.py`` (``STAGE_PARITY_TENSORS``)
