Installation
============

Prerequisites
-------------

- Python 3.10+
- CUDA-capable GPU with a compatible TensorRT / Edge-LLM plugin build
- Hugging Face model access where required

Setup
-----

Install Python dependencies for your environment, then set the Edge-LLM plugin path before
export or TRT inference:

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so

See model-specific pages under :doc:`supported-models` for checkpoint and engine layout notes.
