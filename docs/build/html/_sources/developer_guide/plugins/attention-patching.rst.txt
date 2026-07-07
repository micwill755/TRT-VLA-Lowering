Attention Patching
==================

Patching swaps selected ``self_attn`` modules for plugin-aware wrappers **only on
the export clone**. Original model weights are reused; only the forward path changes.

Helpers in ``trt/plugin/plugin_utils.py``:

- ``patch_vision_attention`` — SigLIP / ViT encoder layers
- ``patch_language_attention`` — decoder LLM layers
- ``restore_attention`` — undo patches after compile


Vision: ``patch_vision_attention``
------------------------------------

.. code-block:: python

   patched = patch_vision_attention(
       vision_model,           # inner SiglipVisionTransformer
       batch_size=B,
       seq_len=S_vit,
       name="vision",
       allow_attention_mask=False,
   )

For each ``vision_model.encoder.layers`` entry:

1. Save ``(layer, layer.self_attn)`` for restore.
2. Replace ``layer.self_attn`` with ``ViTPluginAttention(original_attn, ...)``.

``ViTPluginAttention`` (``attention.py``):

- Reuses ``q_proj``, ``k_proj``, ``v_proj``, ``out_proj`` from the original module.
- Precomputes ``cu_seqlens`` buffer: ``[0, S, 2S, ..., B*S]``.
- Precomputes ``max_seqlen_carrier``: length ``S`` int32 tensor (values unused).
- Reshapes Q/K/V to ``[B*S, num_heads, head_dim]``, casts to fp16.
- Calls ``torch.ops.trt.vit_attention_plugin.default``.
- Reshapes back and applies ``out_proj``.

Set ``allow_attention_mask=True`` only for models that pass a vision attention mask
into the plugin path (most GR00T exports use ``False`` and raise if a mask appears).


Language: ``patch_language_attention``
----------------------------------------

.. code-block:: python

   patched = patch_language_attention(
       language_model,
       hidden_size=H,
       num_attention_heads=N_q,
       num_key_value_heads=N_kv,
       head_dim=D,
       enable_bidirectional_prefill=1,
       name="language",
   )

For each decoder layer:

1. Save original ``self_attn``.
2. Replace with ``PluginAttention`` wrapping projections (and optional Qwen3
   ``q_norm`` / ``k_norm``).

``PluginAttention`` expects runtime/export inputs that the Edge-LLM language runner
supplies:

- ``rope_rotary_cos_sin`` — FP32 external RoPE table
- ``past_key_value`` — KV cache tensor
- ``ctx_len`` — per-batch context lengths
- ``kvcache_start_index`` — cache write positions

The forward path casts Q/K/V to fp16, calls ``torch.ops.trt.attention_plugin.default``,
reshapes output, and applies ``o_proj``.


Restore pattern
---------------

Export hooks **must** restore patches in ``finally`` so cloned modules do not leak
patched attention into later stages or CPU offload paths:

.. code-block:: python

   patched = patch_vision_attention(...)
   try:
       return save_trt_engine_module(...)
   finally:
       restore_attention(patched)

``restore_attention`` reassigns ``layer.self_attn`` from the saved pairs.


Where patching happens
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model / stage
     - Export module
   * - GR00T vision
     - ``trt/executor/models/groot/export/vision.py`` → ``compile()``
   * - GR00T language
     - ``trt/executor/models/groot/export/language.py`` → ``compile()``
   * - Ad-hoc tests
     - ``test_vision.py``, ``molmo2-test.py``


``plan_export`` clones the HuggingFace subgraph inside the stage ``export`` hook
(see export pipeline docs).
Patching operates on ``plan.args["patch_target"]`` or the cloned ``language_model``,
not the live model used for eager inference benchmarks.


Fixed shapes at export time
---------------------------

ViT patching requires explicit ``batch_size`` and ``seq_len`` so buffers like
``cu_seqlens`` are constant in the traced graph. These come from the export plan
(derived from image resolution, patch size, and grid layout).

Language patching uses model config for head counts and head dim; sequence length
remains dynamic within TRT optimization profiles where supported.


Common mistakes
---------------

**Patching the wrong module**

Use the inner ``vision_model.vision_model`` for SigLIP, not the HF wrapper.

**Forgetting restore**

Leaves ``ViTPluginAttention`` on the clone; downstream code may trace zeros.

**Using patched eager for parity**

Patched eager runs stub ops; only TRT engines run real kernels. See
:doc:`parity-and-debugging`.

**Mismatch between patch ``seq_len`` and actual trace**

If the traced forward sees a different token count than ``patch_seq_len``,
``cu_seqlens`` will not match the flattened Q/K/V layout and compile may succeed
with wrong runtime behavior.
