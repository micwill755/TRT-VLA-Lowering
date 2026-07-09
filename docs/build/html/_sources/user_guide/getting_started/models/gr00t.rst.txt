GR00T
=====

**Aliases:** ``gr00t``, ``groot``

NVIDIA GR00T N1.x vision-language-action stack: preprocess, vision, language, action
context, and action stages exported as Edge-LLM–compatible TensorRT engines.

Quick start
-----------

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so

   # Export + benchmark (default)
   python app.py --model gr00t --device cuda --engine-dir /tmp/groot_edge_llm

   # Export only
   python app.py --model gr00t --export-only --device cuda --engine-dir /tmp/groot_edge_llm

   # Benchmark only (requires prior export)
   python app.py --model gr00t --benchmark-only --device cuda --engine-dir /tmp/groot_edge_llm

   # Eager inference only
   python app.py --model gr00t --inference-only --device cuda

Default checkpoint: ``nvidia/GR00T-N1.5-3B``. See :doc:`../../../examples/gr00t` for
the full pipeline walkthrough.
