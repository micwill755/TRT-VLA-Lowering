Alpamayo
========

**Alias:** ``alpamayo``

Alpamayo-R1 vision-language-action model aligned with the Edge-LLM VLA example layout.

Quick start
-----------

Alpamayo is not yet registered in ``app.py``. Use the e2e parity script with the
Alpamayo Python 3.12 environment:

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
   python test_vla_alpamayo_e2e.py

Default checkpoint: ``nvidia/Alpamayo-R1-10B``. See :doc:`../../../examples/alpamayo`
for stage details and the upstream Edge-LLM runtime contract.
