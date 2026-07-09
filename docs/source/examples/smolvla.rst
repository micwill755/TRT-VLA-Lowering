SmolVLA Export Pipeline Example
===============================

**SmolVLA** is a compact vision-language-action model built on SmolVLM2. It shares the
same three-stage Edge-LLM layout as Pi0.5 (vision → language → action).

Quick start
-----------

From the project root:

.. code-block:: bash

   cd /path/to/Test
   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so

Export and benchmark in one command:

.. code-block:: bash

   python app.py --model smolvla --device cuda --engine-dir /tmp/smolvla_edge_llm

Export and benchmark separately:

.. code-block:: bash

   python app.py --model smolvla --export-only --device cuda --engine-dir /tmp/smolvla_edge_llm
   python app.py --model smolvla --benchmark-only --device cuda --engine-dir /tmp/smolvla_edge_llm

Eager inference only:

.. code-block:: bash

   python app.py --model smolvla --inference-only --device cuda

Default checkpoint: ``lerobot/smolvla_base``. Override with ``--model-id``.

Pipeline stages
---------------

.. list-table::
   :header-rows: 1
   :widths: 12 20 24 44

   * - Stage
     - Engine directory
     - Inputs from
     - Responsibility
   * - ``0``
     - ``visual``
     - Runtime inputs
     - SmolVLM2 vision encoder → image embeddings.
   * - ``1``
     - ``language``
     - Stage ``0``
     - Prefix-KV prefill; emits hidden states for the action expert.
   * - ``2``
     - ``action``
     - Stage ``1``
     - Expert denoising loop → robot actions (``final_output``).

.. mermaid::

   graph LR
       IN["runtime sample"]
       V["0 visual"]
       L["1 language"]
       A["2 action"]
       OUT["actions"]

       IN --> V --> L --> A --> OUT

Engine layout
-------------

After export:

.. code-block:: text

   /tmp/smolvla_edge_llm/
     visual/visual.engine
     language/language.engine
       config.json
       embedding.safetensors
       tokenizer artifacts
     action/action.engine

Stage parity script
-------------------

For per-module TRT compile and timing without the full orchestrator, run:

.. code-block:: bash

   python test_vla_smol_e2e.py

*Files:* ``trt/profile/smolvla.py``,
``trt/executor/models/smolvla/export/pipeline.py``,
``trt/executor/models/smolvla/inference/pipeline.py``
