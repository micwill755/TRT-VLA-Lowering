GR00T Language Export Example
=============================

This page walks through the **language stage**: Eagle Qwen decoder prefill compiled
as ``language/language.engine`` for the Edge-LLM ``LLMEngineRunner``. Vision
embeddings are spliced into ``inputs_embeds`` before trace.


Typical tensor shapes (GR00T N1.5, Libero sample)
-------------------------------------------------

Sequence length ``S`` is dynamic (packed prompt after image splice; often ~566).

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Tensor
     - Shape
     - Notes
   * - ``image_embs`` (upstream)
     - ``[512, 2048]``
     - Flat vision rows from stage 0
   * - ``inputs_embeds``
     - ``[1, S, 2048]`` fp16
     - Text embeddings with vision rows at ``image_token_id`` slots
   * - ``rope_rotary_cos_sin``
     - ``[1, S, ...]`` fp16
     - External RoPE table for plugin attention
   * - ``context_lengths``
     - ``[1]`` int32
     - Valid prefill length per batch row
   * - ``kvcache_start_index``
     - ``[0]`` int32
     - Empty for fresh prefill
   * - ``last_token_ids``
     - ``[1, 1]`` int64
     - Index of last token for logits gather
   * - ``past_key_values_i``
     - ``[1, 2, KvH, S, D]`` fp16 per layer
     - Prefill KV scratch; second dim is key/value
   * - ``lm_hidden_states`` (output)
     - ``[1, S, 2048]`` fp16
     - Final RMSNorm hidden — fed to action_context
   * - ``logits`` (output)
     - ``[1, V]`` fp32
     - Last-token logits (Edge-LLM contract)


Export module wiring
--------------------

**Preprocess** splices vision into token embeddings and builds flat Edge-LLM bindings:

.. code-block:: python

   # Splice vision rows into image placeholder positions
   input_embs = language.get_input_embeddings()(input_ids)
   flat_embs[image_token_mask] = flat_image_embs[:num_slots]
   inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden).contiguous()

   lm_module = CausalLMExportModule(
       decoder,                    # Qwen3Model.layers
       language.lm_head,
       select_layer=-1,             # final norm hidden for context
   ).eval().to(device=device)

   lm_inputs = (
       trt_inputs_embeds,           # fp16 spliced embeds
       rope_rotary_cos_sin,
       ctx_len,
       kvcache_start_index,
       last_token_ids,
       *kv_caches,                  # one tensor per decoder layer
   )

**Export** patches language attention and writes the engine plus sidecars:

.. code-block:: python

   patched = patch_language_attention(decoder, hidden_size=..., ...)
   try:
       engine_path = save_trt_engine_module(
           lm_module,
           lm_inputs,
           ctx.engine_root / "language",
           engine_file="language.engine",
           input_names=["inputs_embeds", "rope_rotary_cos_sin", ...],
           output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
           trt_settings={**ctx.trt_settings, **language_edge_trt_settings()},
       )
   finally:
       restore_attention(patched)

``save_artifacts`` writes ``embedding.safetensors``, tokenizer JSON, and chat
template beside the engine for the C++ runtime.


Putting it together — manual decoder loop
-----------------------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       IN["inputs_embeds [B,S,H]<br/>+ RoPE + KV bindings"]

       LOOP["for each decoder layer"]
       ATTN["PluginAttention<br/>self_attn + KV update"]
       MLP["post-attn MLP"]
       NORM["final RMSNorm"]

       LOG["lm_head → logits [B,V]"]
       CTX["context_hidden → lm_hidden_states [B,S,H]"]
       PKV["stack prefix_k / prefix_v"]

       IN --> LOOP --> ATTN --> MLP --> LOOP
       LOOP --> NORM
       NORM --> LOG
       NORM --> CTX
       LOOP --> PKV

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class ATTN nvNode
       class IN,LOOP,MLP,NORM,LOG,CTX,PKV greyNode


Step-by-step
~~~~~~~~~~~~

**1. Vision splice (preprocess, not inside engine)**

Text token embeddings are built from ``input_ids``. Rows where
``input_ids == image_token_id`` are replaced with vision ``image_embs`` rows.
This matches eager Eagle forward and the C++ ``embeddingLookupWithImageInsertion``
path.

**2. Manual layer loop**

:class:`CausalLMExportModule` does **not** call ``language_model.forward()``. It
iterates ``decoder.layers`` explicitly so each ``self_attn`` can be replaced with
``PluginAttention`` during compile.

**3. Outputs**

- ``logits`` — gathered at ``last_token_ids`` for standard LM use.
- ``lm_hidden_states`` — full-sequence hidden after final norm; GR00T action
  pipeline uses this for action_context.
- ``prefix_k`` / ``prefix_v`` — stacked per-layer KV for optional prefix-cache
  consumers (Pi0.5 / SmolVLA paths).


Eager vs TRT inference note
---------------------------

At **inference** time, ``ExecutionMode.EAGER`` uses the stock HuggingFace
``language_model`` forward (model weight dtype). ``IN_MEMORY`` and ``SERIALIZED``
use :class:`CausalLMExportModule` or the flat serialized ABI respectively. Export
always traces ``CausalLMExportModule``.


Dummy tensor for downstream export
----------------------------------

Language export returns a **zero** ``lm_hidden`` tensor with shape
``[B, S, H]`` so action_context can trace without re-running the language engine
immediately after compile.


Parity tips
-----------

1. **Same splice inputs** — parity failures often come from mismatched
   ``image_embs`` or wrong ``image_token_id`` slot count vs ``seq_len_per_image``.
2. **Dtype** — TRT bindings use ``ctx.dtype`` (fp16); eager reference uses
   ``lm_dtype = next(language.parameters()).dtype``.
3. Benchmark parity key: ``language:lm_hidden``.


*Files:* ``trt/modules/export/language.py``, ``trt/executor/models/groot/export/language.py``,
``trt/language.py``, ``trt/rope.py``, ``trt/tokenizer.py``
