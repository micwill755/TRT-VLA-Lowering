Custom Ops
==========

Custom ops are the **stable names** Torch-TensorRT pattern-matches during compile.
They sit between the attention wrappers and the TRT plugin layer.

Registration uses ``torch.library.custom_op`` in ``plugin_utils.py``. Each op has:

- A **real implementation** used during ``torch.export`` tracing.
- A **fake** (meta) implementation registered with ``@op.register_fake`` for
  shape inference and ``torch.compile`` / Dynamo analysis.


Registered ops
--------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Op name
     - Called from
   * - ``trt::attention_plugin``
     - ``PluginAttention.forward`` in ``attention.py``
   * - ``trt::vit_attention_plugin``
     - ``ViTPluginAttention.forward`` in ``attention.py``


Python access:

.. code-block:: python

   torch.ops.trt.attention_plugin.default(...)
   torch.ops.trt.vit_attention_plugin.default(...)


LLM attention op
----------------

Signature (simplified):

.. code-block:: python

   @torch.library.custom_op("trt::attention_plugin", mutates_args=())
   def attention_plugin(q, k, v, past_key_value, context_lengths,
                        rope_rotary_cos_sin, kvcache_start_index,
                        num_q_heads, num_kv_heads, enable_tree_attention,
                        head_size, enable_fp8_kv_cache, ...) -> Tuple[Tensor, Tensor]:
       ...
       return attn_output, past_key_value.clone()

**Real body behavior:** allocates ``attn_output`` as **zeros** with shape
``[batch, seq_len, num_q_heads, head_size]`` and returns the KV tensor unchanged
(cloned). Projections and RoPE happened in ``PluginAttention`` before the op call.

**Fake body behavior:** returns ``empty`` tensors with the same shapes for meta
propagation.


ViT attention op
----------------

.. code-block:: python

   @torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
   def vit_attention_plugin(q, k, v, cu_seqlens, max_seqlen_carrier,
                            num_heads, head_size) -> Tensor:
       return torch.zeros_like(q)

**Real body behavior:** returns **zeros** with the same shape/dtype/device as ``q``.

**Fake body behavior:** ``torch.empty_like(q)``.

Input layout from ``ViTPluginAttention``:

- ``q``, ``k``, ``v``: ``[batch * seq_len, num_heads, head_dim]``, fp16, contiguous
- ``cu_seqlens``: cumulative sequence offsets, length ``batch + 1``
- ``max_seqlen_carrier``: int32 tensor whose **length** encodes max sequence length
  (values are unused; plugin reads the tensor extent)


Why placeholders are correct
----------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000'}}}%%
   graph LR
       TRACE[torch.export trace] --> FX[FX graph with custom op]
       FX --> DYNAMO[Torch-TensorRT Dynamo]
       DYNAMO --> REPLACE[Op replaced by TRT plugin]
       REPLACE --> ENG[Serialized engine]
       ENG --> KERNEL[Real fused kernel at runtime]

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class DYNAMO,REPLACE,KERNEL nvNode
       class TRACE,FX,ENG greyNode

During export, PyTorch only needs valid tensor **shapes** and **dtypes** to build
the graph. The custom op body can be a stub because:

1. ``register_fake`` supplies shape metadata to the compiler.
2. The Dynamo **converter** (see :doc:`converters`) never executes the Python body;
   it emits a TRT plugin node instead.
3. At inference, the engine executes the C++ plugin implementation.

Running the patched module eagerly in PyTorch **will** produce garbage attention
output (zeros). That is by design.


Interaction with edge_plugins
-----------------------------

When ``torch_tensorrt.dynamo.conversion.edge_plugins`` is available,
``load_edge_plugin`` may register ``trt::attention_plugin`` before this project
does. ``_register_attention_plugin_op`` skips re-registration in that case.

``trt::vit_attention_plugin`` is always registered here because the stock Edge-LLM
Python bindings do not cover the ViT export path used by GR00T vision.


Adding a new custom op
----------------------

1. Define ``@torch.library.custom_op("trt::your_op", ...)`` with a minimal stub body.
2. Add ``@your_op.register_fake`` returning correctly shaped ``empty`` tensors.
3. Call the registration function from ``load_plugins_for_trt()``.
4. Implement a ``@dynamo_tensorrt_converter`` in ``plugin_converter.py``.
5. Call the op from a patched ``nn.Module`` so ``torch.export`` records it.

Keep scalar hyperparameters (head counts, flags) as op arguments or converter
kwargs so they become ``PluginField`` values on the TRT side.
