Architecture
============

The plugin stack has four cooperating layers. Each layer has a single job; together
they bridge HuggingFace attention modules and fused Edge-LLM kernels.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       subgraph PY ["Python (export only)"]
           ATTN[attention.py<br/>PluginAttention / ViTPluginAttention]
           UTILS[plugin_utils.py<br/>patch + register ops + load .so]
           CONV[plugin_converter.py<br/>Dynamo converters]
       end

       subgraph PT ["PyTorch dispatcher"]
           COP[torch.ops.trt.attention_plugin]
           VOP[torch.ops.trt.vit_attention_plugin]
       end

       subgraph TRT ["TensorRT build"]
           CREATOR[IPluginCreator from .so]
           LAYER[IPluginV2Layer in engine graph]
       end

       subgraph RT ["Runtime"]
           KERNEL[Fused MHA in libNvInfer_edgellm_plugin.so]
       end

       ATTN --> COP
       ATTN --> VOP
       UTILS --> COP
       UTILS --> VOP
       COP --> CONV
       VOP --> CONV
       CONV --> CREATOR --> LAYER --> KERNEL

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef lightSubGraph fill:none,stroke:#aaa,stroke-width:1.5px

       class ATTN,UTILS,CONV nvNode
       class COP,VOP,CREATOR,LAYER,KERNEL greyNode
       class PY,PT,TRT,RT lightSubGraph


Layer responsibilities
----------------------

**Attention wrappers** (``trt/plugin/attention.py``)
   Replace ``nn.Module`` attention layers during export. They keep the original
   ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` weights and call a custom op
   for the attention math.

**Plugin utilities** (``trt/plugin/plugin_utils.py``)
   Register custom ops if Torch-TensorRT's ``edge_plugins`` package did not already
   register them, load the Edge-LLM ``.so``, patch/restore attention modules, and
   expose ``get_trt_plugin_creator`` for converters.

**Dynamo converters** (``trt/plugin/plugin_converter.py``)
   Map each custom op occurrence in the exported FX graph to a TensorRT
   ``add_plugin_v2`` node with the correct ``PluginField`` attributes.

**Edge-LLM shared library**
   Provides ``AttentionPlugin`` and ``ViTAttentionPlugin`` creators and the fused
   GPU kernels executed when a serialized engine runs.


Compile-shim pattern
--------------------

The wrappers are intentionally **not** numerically faithful in eager PyTorch:

- ViT custom op returns ``torch.zeros_like(q)`` during trace/eager.
- LLM custom op returns zeros for attention output and clones the KV cache tensor.

That is expected. The real math runs only after Torch-TensorRT replaces the op
with the TRT plugin during compile, and again when the ``.engine`` executes.

.. important::

   Do **not** use patched eager output as a parity baseline. Compare **unpatched
   eager** against **serialized TRT** (see :doc:`parity-and-debugging`).


End-to-end export sequence
--------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   sequenceDiagram
       participant Orch as EdgeOrchestrator
       participant Utils as plugin_utils
       participant Export as ExportRunner
       participant Patch as patch_*_attention
       participant TRT as torch_tensorrt.dynamo

       Orch->>Utils: load_plugins_for_trt()
       Utils->>Utils: register custom ops
       Utils->>Utils: ctypes / edge_plugins load .so
       Utils->>Utils: import plugin_converter
       Export->>Patch: swap attention on cloned module
       Export->>TRT: torch.export + compile
       TRT->>TRT: converter matches custom op
       TRT-->>Export: engine with plugin layers
       Export->>Patch: restore_attention (finally)


SigLIP nesting (GR00T vision)
-----------------------------

HuggingFace wraps the actual ViT transformer one level above where attention
lives:

.. code-block:: text

   eagle_model.vision_model              # SiglipVisionModel (HF wrapper)
   eagle_model.vision_model.vision_model # SiglipVisionTransformer
     ├── embeddings
     ├── encoder.layers[i].self_attn     ← patch target
     └── post_layernorm

Export plans pass ``patch_target`` pointing at the inner ``vision_model`` so
``patch_vision_attention`` replaces the layers that ``torch.export`` actually
traces.


Key files
---------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Role
   * - ``trt/plugin/attention.py``
     - ``PluginAttention``, ``ViTPluginAttention`` forward paths
   * - ``trt/plugin/plugin_utils.py``
     - Op registration, ``load_plugins_for_trt``, patch helpers
   * - ``trt/plugin/plugin_converter.py``
     - ``convert_llm_attention_plugin``, ``convert_vit_attention_plugin``
   * - ``trt/orchestrator/edge_orchestrator.py``
     - Calls ``load_plugins_for_trt()`` at pipeline start
   * - ``trt/executor/models/groot/export/vision.py``
     - Vision export hook using ``patch_vision_attention``
   * - ``trt/executor/models/groot/export/language.py``
     - Language export hook using ``patch_language_attention``
