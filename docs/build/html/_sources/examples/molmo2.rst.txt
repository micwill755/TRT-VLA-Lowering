MolmoAct2 Export Pipeline Example
=================================

**MolmoAct2** (CLI alias ``molmo2``) is Allen AI's multimodal action model. It uses a
token-pooling vision backbone and a split language path (encoder prefill + discrete
causal LM) before the continuous flow-matching action head.

Quick start
-----------

MolmoAct2 has a registered profile (``--model molmo2``) but the full export/inference
pipeline is not yet wired into ``app.py``. Use the end-to-end parity script today:

.. code-block:: bash

   cd /path/to/Test
   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
   python test_vla_molmo2_e2e.py

Default checkpoint: ``allenai/MolmoAct2``.

When the orchestrator pipeline is registered, the intended ``app.py`` usage will be:

.. code-block:: bash

   python app.py --model molmo2 --device cuda --engine-dir /tmp/molmoact2_edge_llm

Pipeline stages (e2e script)
----------------------------

The e2e script exports and benchmarks four logical boundaries:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Module
     - Export wrapper
     - Responsibility
   * - Vision
     - ``TokenPoolingExportModule``
     - Token-pooling vision backbone → pooled image tokens.
   * - Language encoder
     - ``MolmoTextEncoderKVExportModule``
     - Prefill encoder; emits hidden states and prefix KV.
   * - Discrete LM
     - ``MolmoTextCausalLMExportModule``
     - Causal LM head for discrete action tokens.
   * - Action
     - ``MolmoAct2ActionFlowStepExportModule``
     - Flow-matching velocity step for continuous actions.

.. mermaid::

   graph LR
       IN["runtime sample"]
       V["vision"]
       ENC["language encoder"]
       DISC["discrete LM"]
       A["action"]
       OUT["actions"]

       IN --> V --> ENC --> DISC --> A --> OUT

Engine layout (planned)
-----------------------

Target layout under ``/tmp/molmoact2_edge_llm/``:

.. code-block:: text

   /tmp/molmoact2_edge_llm/
     visual/visual.engine
     language/language.engine
     action/action.engine

*Files:* ``trt/profile/molmoact2.py``, ``test_vla_molmo2_e2e.py``,
``trt/modules/export/vision.py``, ``trt/modules/export/language.py``,
``trt/modules/export/diffusion.py``
