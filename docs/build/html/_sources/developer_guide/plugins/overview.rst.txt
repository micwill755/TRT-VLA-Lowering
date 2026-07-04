Plugins Overview
================

Torch-TRT pipelines lowers attention-heavy subgraphs to **TensorRT Edge-LLM**
plugins instead of decomposing multi-head attention into hundreds of native TRT
layers. The Python code under ``trt/plugin/`` is a **compile shim**: it makes the
export graph contain recognizable custom ops that Torch-TensorRT can lower to
fused C++ kernels in ``libNvInfer_edgellm_plugin.so``.

At runtime, inference does **not** execute the Python shim. Serialized engines
call the plugin directly. The shim exists only so ``torch.export`` and
``torch_tensorrt.dynamo.compile`` can capture the right graph structure.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       EAGER[Eager PyTorch model]
       PATCH[patch_*_attention]
       SHIM[ViTPluginAttention / PluginAttention]
       OP[torch.ops.trt.*_plugin]
       CONV[Dynamo converter]
       TRT[TensorRT plugin layer]
       ENG[.engine file]
       RT[Edge-LLM runtime]

       EAGER --> PATCH --> SHIM --> OP --> CONV --> TRT --> ENG --> RT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class PATCH,CONV,TRT nvNode
       class EAGER,SHIM,OP,ENG,RT greyNode


Two plugin families
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Use case
     - Python module
     - TRT plugin / custom op
   * - Vision (SigLIP / ViT)
     - ``ViTPluginAttention``
     - ``ViTAttentionPlugin`` / ``trt::vit_attention_plugin``
   * - Language decoder (LLM)
     - ``PluginAttention``
     - ``AttentionPlugin`` / ``trt::attention_plugin``


When plugins load
-----------------

``load_plugins_for_trt()`` runs at orchestrator startup (``EdgeOrchestrator.run``)
and before standalone export scripts compile engines. It must run **before**
``torch_tensorrt.dynamo.compile`` so that:

1. Custom ops are registered in the PyTorch dispatcher.
2. The Edge-LLM shared library is loaded into the TRT plugin registry.
3. Dynamo converters in ``plugin_converter.py`` are imported and registered.

See :doc:`registration` for the step-by-step sequence.


Export integration
------------------

Model export hooks patch attention on a **cloned** subgraph, compile, then restore
the original modules in a ``finally`` block. For GR00T vision:

.. code-block:: python

   patched = patch_vision_attention(patch_target, batch_size=..., seq_len=..., name="vision")
   try:
       save_trt_engine_module(plan.module, ...)
   finally:
       restore_attention(patched)

The patch target for SigLIP is the **inner** transformer
(``vision_model.vision_model``), not the HuggingFace wrapper. Patching the outer
module leaves ``encoder.layers[i].self_attn`` unchanged and the TRT graph will
not contain the ViT plugin op.

Related pages
-------------

- :doc:`architecture` — compile-shim layers and data flow
- :doc:`registration` — ``load_plugins_for_trt`` and environment setup
- :doc:`custom-ops` — ``torch.library.custom_op`` and fake implementations
- :doc:`converters` — Torch-TensorRT lowering to TRT plugin layers
- :doc:`attention-patching` — ``ViTPluginAttention`` vs ``PluginAttention``
- :doc:`parity-and-debugging` — what to compare when validating engines

Upstream Edge-LLM reference:
`TensorRT Plugins Guide
<https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/plugins-guide.html>`_.
