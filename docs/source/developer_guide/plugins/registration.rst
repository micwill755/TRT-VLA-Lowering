Plugin Registration
===================

Registration wires three things together before any engine compile:

1. **PyTorch custom ops** — so ``torch.export`` can record ``trt::attention_plugin``
   and ``trt::vit_attention_plugin`` nodes in the graph.
2. **TensorRT plugin library** — so ``IPluginCreator`` for ``AttentionPlugin`` and
   ``ViTAttentionPlugin`` exists at compile time.
3. **Torch-TensorRT converters** — so Dynamo knows how to lower those ops to TRT
   plugin layers.

Entry point: ``load_plugins_for_trt()``
---------------------------------------

Defined in ``trt/plugin/plugin_utils.py``:

.. code-block:: python

   def load_plugins_for_trt():
       _register_attention_plugin_op()
       _register_vit_attention_plugin_op()
       load_plugin()
       from trt.plugin import plugin_converter as _plugin_converter  # noqa: F401


Call order matters:

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000'}}}%%
   graph TD
       A[_register_attention_plugin_op] --> B[_register_vit_attention_plugin_op]
       B --> C[load_plugin]
       C --> D[import plugin_converter]

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       class A,B,C,D nvNode


Converters must be imported **after** the ``.so`` is loaded so
``get_trt_plugin_creator`` can find the plugin creators when compile runs.


When it runs
------------

``load_plugins_for_trt()`` is invoked from:

- ``EdgeOrchestrator.run`` — default export/benchmark CLI path
- ``trt/serialize.py`` — standalone engine serialization helpers
- Test scripts such as ``test_vision.py`` before manual compile

If you add a new export entry point, call ``load_plugins_for_trt()`` once at
startup, before the first ``torch_tensorrt.dynamo.compile``.


Loading the shared library
--------------------------

``load_plugin()`` resolves the plugin path from the environment:

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so

``EDGELLM_TRT_PLUGIN_SO`` is accepted as an alias.

Loading strategy:

1. Try ``torch_tensorrt.dynamo.conversion.edge_plugins.load_edge_plugin(plugin_so)``
   when Torch-TensorRT was built with edge plugin support (registers stock ops).
2. On ``ImportError``, fall back to ``ctypes.CDLL(plugin_so)``.
3. Always call ``trt.init_libnvinfer_plugins(None, "")`` so creators are visible
   to ``get_trt_plugin_creator``.

If the environment variable is unset, ``load_plugin()`` raises immediately rather
than producing a graph with missing plugins at compile time.


Idempotent op registration
--------------------------

``_register_attention_plugin_op`` and ``_register_vit_attention_plugin_op`` check
whether the op already exists:

.. code-block:: python

   if _has_torch_op("trt", "vit_attention_plugin"):
       return

When Torch-TensorRT's ``edge_plugins`` package is present, it may register
``trt::attention_plugin`` first. This project still registers
``trt::vit_attention_plugin`` locally because ViT lowering is pipeline-specific.


Plugin creator lookup (TRT 10 vs 11)
------------------------------------

``get_trt_plugin_creator`` abstracts API differences:

- TensorRT 10: ``registry.get_plugin_creator(name, version, namespace)``
- TensorRT 11+: ``registry.get_creator(name, version, namespace)``

Converters call this helper rather than touching the registry directly.


Configuration hook
------------------

``get_plugin_config()`` returns a module-level dict used by the LLM attention
converter. Export code can set ``enable_bidirectional_prefill`` (and future
fields) before compile so the converter passes the right ``PluginField`` values
to ``AttentionPlugin``.

Vision export does not use this dict today; ViT plugin fields come entirely from
tensor shapes and scalar arguments in the custom op call.


Checklist for new environments
-------------------------------

1. Build or obtain ``libNvInfer_edgellm_plugin.so`` for your TensorRT version.
2. Set ``EDGE_LLM_PLUGIN_SO`` in the shell or CI job.
3. Ensure ``load_plugins_for_trt()`` runs before compile.
4. Confirm creators resolve:

   .. code-block:: python

      from trt.plugin.plugin_utils import load_plugins_for_trt, get_trt_plugin_creator
      load_plugins_for_trt()
      assert get_trt_plugin_creator("ViTAttentionPlugin", "1", "") is not None
