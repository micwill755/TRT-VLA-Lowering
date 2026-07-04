Converters
==========

Converters teach **Torch-TensorRT Dynamo** how to turn a custom PyTorch op into a
TensorRT ``IPluginV2`` layer. They live in ``trt/plugin/plugin_converter.py`` and
register at import time via ``@dynamo_tensorrt_converter``.

.. code-block:: python

   @dynamo_tensorrt_converter(
       torch.ops.trt.vit_attention_plugin.default,
       supports_dynamic_shapes=True,
       priority=ConverterPriority.HIGH,
   )
   def convert_vit_attention_plugin(ctx, target, args, kwargs, name):
       ...


``load_plugins_for_trt()`` imports this module **last** so decorators run after the
``.so`` is loaded and custom ops exist.


Converter responsibilities
--------------------------

For each matched op in the exported graph, a converter must:

1. **Unpack arguments** — Q/K/V tensors, buffers, and scalar hyperparameters.
2. **Look up** ``IPluginCreator`` via ``get_trt_plugin_creator(plugin_name, "1", "")``.
3. **Build** ``PluginFieldCollection`` with INT32/FLOAT32 fields the C++ plugin expects.
4. **Wrap** input tensors with ``get_trt_tensor`` when they are still Torch tensors.
5. **Add** ``ctx.net.add_plugin_v2(inputs, plugin)`` and return output TRT tensor(s).


ViT converter
-------------

``convert_vit_attention_plugin`` targets ``ViTAttentionPlugin``:

**Inputs (TRT tensors)**

- ``q``, ``k``, ``v``
- ``cu_seqlens``
- ``max_seqlen_carrier``

**Plugin fields**

- ``num_heads`` (INT32)
- ``head_size`` (INT32)

**Output**

- Single attention output tensor (same rank as ``q``)

.. code-block:: python

   creator = get_trt_plugin_creator("ViTAttentionPlugin", "1", "")
   plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
   layer = ctx.net.add_plugin_v2(inputs, plugin)
   return layer.get_output(0)


LLM converter
-------------

``convert_llm_attention_plugin`` targets ``AttentionPlugin``. It overrides the
stock Torch-TensorRT edge converter to:

- Pass **separate** Q/K/V tensors (no fused QKV slice).
- Honor ``enable_bidirectional_prefill`` from ``get_plugin_config()``.

**Inputs**

- ``q``, ``k``, ``v``
- ``past_key_value`` (KV cache)
- ``context_lengths``
- ``rope_rotary_cos_sin`` (FP32)
- ``kvcache_start_index``
- Optional tree-attention tensors when enabled

**Plugin fields**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Source
   * - ``num_q_heads``, ``num_kv_heads``, ``head_size``
     - Scalar op arguments
   * - ``enable_tree_attention``, ``enable_fp8_kv_cache``
     - Scalar op arguments
   * - ``sliding_window_size``
     - Scalar op argument (default ``-1``)
   * - ``enable_bidirectional_prefill``
     - ``get_plugin_config()`` (GR00T language export)
   * - ``qkv_scales``
     - Optional FLOAT32 array when FP8 KV cache is enabled

**Outputs**

- Attention output tensor
- Updated KV cache tensor

The converter includes a small shape fix: if ``kvcache_start_index`` arrives as
``[batch, 1]``, a shuffle layer reshapes it to ``[batch]`` before the plugin sees it.


Priority and dynamic shapes
---------------------------

Both converters set:

- ``supports_dynamic_shapes=True`` — batch/sequence dims can vary within TRT profile bounds.
- ``priority=ConverterPriority.HIGH`` — prefer this converter over stock edge converters
  when multiple handlers match.


Compile-time flow
-----------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000'}}}%%
   sequenceDiagram
       participant FX as Exported FX graph
       participant Reg as ConverterRegistry
       participant Conv as plugin_converter
       participant Net as TRT NetworkDefinition
       participant SO as libNvInfer_edgellm_plugin.so

       FX->>Reg: vit_attention_plugin node
       Reg->>Conv: convert_vit_attention_plugin
       Conv->>SO: get_trt_plugin_creator("ViTAttentionPlugin")
       Conv->>Net: add_plugin_v2(q,k,v,...)
       Net-->>FX: TRT tensor output


Debugging converter issues
--------------------------

**"Plugin not found in TensorRT plugin registry"**

- ``EDGE_LLM_PLUGIN_SO`` not set or wrong path.
- ``trt.init_libnvinfer_plugins`` not called before compile.
- TensorRT major version mismatch between plugin build and runtime.

**Converter never runs / graph still has native attention ops**

- Attention layers were not patched before ``torch.export``.
- Wrong SigLIP module patched (outer wrapper instead of inner ``vision_model``).
- ``load_plugins_for_trt()`` not called; ``plugin_converter`` never imported.

**"Registering converter" log spam**

- Debug builds of Torch-TensorRT log each registration at import. Use
  ``torch_tensorrt.logging.set_level(logging.ERROR)`` where available, or reduce
  log noise at import time in test scripts.


Extending converters
--------------------

When the C++ plugin gains new ``PluginField`` entries:

1. Thread the value through the custom op signature or ``get_plugin_config()``.
2. Append a ``trt.PluginField`` in the converter's ``field_list``.
3. Re-export engines — existing ``.engine`` files are incompatible with changed
   plugin field layouts unless the plugin version bumps and you rebuild.

Keep converter logic thin: shape tweaks and field packing only; attention math stays
in the Edge-LLM plugin.
