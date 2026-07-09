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

Edge LLM runtime
----------------

SmolVLA shares the Pi0.5 three-engine layout and ``prefix_k`` / ``prefix_v``
handoff. Vision uses **64** connector tokens per image (not 1024 raw SigLIP
patches) and ``image_token_id`` for ``<image>`` (49190).

.. code-block:: bash

   export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so

   llm_inference \
     --engineDir=/tmp/smolvla_edge_llm/language \
     --multimodalEngineDir=/tmp/smolvla_edge_llm \
     --inputFile=/tmp/smolvla_edge_llm/runtime_smoke/input_action.json \
     --outputFile=/tmp/smolvla_edge_llm/runtime_smoke/output_e2e.json \
     --maxGenerateLength=0

Default export uses a static prefix of **151** tokens (2×64 image + language +
state). Action ``attention_mask`` must be float32 at the engine boundary. See
:doc:`../edge_llm/e2e` for the full checklist and common failures.

Full binding contract: :doc:`../edge_llm/overview`, :doc:`../edge_llm/runners`,
:doc:`../edge_llm/bindings`.
