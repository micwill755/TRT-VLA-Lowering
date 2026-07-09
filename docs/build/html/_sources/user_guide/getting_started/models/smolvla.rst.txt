SmolVLA
=======

**Alias:** ``smolvla``

SmolVLA compact vision-language-action model with three Edge-LLM engines
(vision, language, action).

Quick start
-----------

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so

   # Export + benchmark (default)
   python app.py --model smolvla --device cuda --engine-dir /tmp/smolvla_edge_llm

   # Export only
   python app.py --model smolvla --export-only --device cuda --engine-dir /tmp/smolvla_edge_llm

   # Benchmark only (requires prior export)
   python app.py --model smolvla --benchmark-only --device cuda --engine-dir /tmp/smolvla_edge_llm

   # Eager inference only
   python app.py --model smolvla --inference-only --device cuda

Default checkpoint: ``lerobot/smolvla_base``. See :doc:`../../../examples/smolvla` for
stage details and the e2e parity script.
