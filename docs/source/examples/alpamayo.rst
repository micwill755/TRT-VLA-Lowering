Alpamayo Export Pipeline Example
================================

**Alpamayo-R1** is NVIDIA's Qwen3-VL–based vision-language-action model for
autonomous driving. It follows the same three-engine Edge-LLM layout as Pi0.5
(vision → language → action).

Quick start
-----------

Alpamayo is not yet registered in ``app.py``. Run the end-to-end parity script
with the Alpamayo Python 3.12 environment (requires ``hydra-core`` and the
``alpamayo`` package):

.. code-block:: bash

   cd /path/to/Test
   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
   python test_vla_alpamayo_e2e.py

Default checkpoint: ``nvidia/Alpamayo-R1-10B``.

The script loads a Physical AI AV dataset clip for sample images and text. See the
`upstream Alpamayo example
<https://nvidia.github.io/TensorRT-Edge-LLM/latest/examples/alpamayo.html>`_ for
the C++ runtime contract.

Pipeline stages (e2e script)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 28 52

   * - Module
     - Export wrapper
     - Responsibility
   * - Vision
     - ``VisualFixedGrid``
     - Qwen3-VL vision tower with patched attention.
   * - Language
     - ``Qwen3VLTextModelPrefillExportModule``
     - VLM prefill; emits hidden states and prefix KV.
   * - Action
     - ``StaticActionVelocityStepExportModule``
     - Expert flow-matching denoising step.

.. mermaid::

   graph LR
       IN["AV clip frames + prompt"]
       V["vision"]
       L["language"]
       A["action"]
       OUT["trajectory actions"]

       IN --> V --> L --> A --> OUT

Engine layout (planned)
-----------------------

Target layout aligned with Edge-LLM VLA runtime:

.. code-block:: text

   /tmp/alpamayo_edge_llm/
     visual/visual.engine
     language/language.engine
       config.json
       embedding.safetensors
       tokenizer artifacts
     action/action.engine

*Files:* ``test_vla_alpamayo_e2e.py``,
``trt/modules/export/alpamayo_vision.py``,
``trt/modules/export/alpamayo_language.py``,
``trt/modules/export/diffusion.py``
