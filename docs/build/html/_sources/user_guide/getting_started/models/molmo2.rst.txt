Molmo2
======

**Alias:** ``molmo2``

MolmoAct2 multimodal model pipeline for vision-language export and inference.

Quick start
-----------

The full orchestrator pipeline is in progress. Use the e2e parity script today:

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
   python test_vla_molmo2_e2e.py

When registered, ``app.py`` will support:

.. code-block:: bash

   python app.py --model molmo2 --device cuda --engine-dir /tmp/molmoact2_edge_llm

Default checkpoint: ``allenai/MolmoAct2``. See :doc:`../../../examples/molmo2` for
module boundaries and engine layout.
