Binding Contract: config.json to GPU Memory
===========================================

This is the exact mechanism the question is about: how the ``config.json`` written
during export tells ``LLMInferenceRuntime`` what GPU tensors to allocate and which
engine bindings to attach them to.

Two things must line up:

1. **Binding names** — the string names of the engine's input/output tensors.
2. **Config numbers** — the shapes/flags the runtime uses to size buffers.

Export writes both; the runtime reads both.

Step 1 — Export bakes binding names into the engine and config.json
-------------------------------------------------------------------

Every engine is written by :func:`save_trt_engine_module`
(``trt/compile.py``). It compiles the traced module, forces the output binding
names, and writes a ``config.json`` beside the ``.engine``:

.. code-block:: python

   save_trt_engine_module(
       module,
       sample_inputs,
       ctx.engine_root / "visual",
       engine_file="visual.engine",
       model_type="visual",
       component="vision",
       input_names=["pixel_values"],       # engine input binding
       output_names=["visual_embeds"],     # engine output binding
       extra_config={
           "vocab_size": vocab_size,
           "image_token_id": image_token_id,
           "seq_len": seq_len,
       },
       trt_settings=ctx.trt_settings,
   )

Internally it records the names and per-tensor metadata into ``config.json``:

.. code-block:: python

   config = {
       "model_type": model_type,
       "component": component,
       "engine_file": engine_file,
       "precision": "FP16",
       "input_names": list(input_names),
       "inputs":  {name: tensor_meta(t) for name, t in zip(input_names, flat_tensors)},
       "outputs": [tensor_meta(t) for t in flatten_tensors(example_output)],
   }
   if output_names:
       config["output_names"] = list(output_names)
   if extra_config:
       config.update(extra_config)

The output binding names are applied to the TRT engine itself via
``patch_trt_interpreter_output_names`` so the compiled engine advertises exactly
``logits`` / ``context_embs`` / ``visual_embeds`` / etc. Those strings are the
**same constants** the C++ side hard-codes in ``common/bindingNames.h``:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Engine
     - Input bindings
     - Output bindings
   * - visual
     - ``pixel_values`` (HWC)
     - ``visual_embeds``
   * - language
     - ``inputs_embeds``, ``rope_rotary_cos_sin``, ``context_lengths``,
       ``kvcache_start_index``, ``last_token_ids``, ``past_key_values_{i}``
     - ``logits`` (+ optional ``lm_hidden_states`` / ``context_embs`` /
       ``prefix_k`` / ``prefix_v``)
   * - action_context
     - ``lm_hidden_states``
     - ``vl_embs``
   * - action (GR00T)
     - ``actions``, ``timestep``, ``context_embs``, ``state``, ``embodiment_id``
     - ``velocity``
   * - action (Pi0.5)
     - ``x_t``, ``timestep``, ``prefix_k``, ``prefix_v``, ``position_ids``,
       ``attention_mask``
     - ``velocity``

On the Python side these names are centralized in ``trt/io_spec.py``
(``PipelineIOSpec`` / ``GROOT_EDGE_IO`` / ``PI05_EDGE_IO``), so the export modules
and the C++ ``bindingNames.h`` describe the same tensors.

Step 2 — Language config.json carries the sizing numbers
--------------------------------------------------------

The language engine's ``config.json`` is special: it drives almost every buffer in
the runtime. It is built by ``language_edge_llm_config`` (``trt/language.py``):

.. code-block:: python

   edge_config = {
       "vocab_size": ...,
       "max_position_embeddings": ...,
       "hidden_size": ...,
       "num_hidden_layers": ...,
       "num_attention_heads": ...,
       "num_key_value_heads": ...,
       "head_dim": ...,
       "builder_config": {
           "max_batch_size": batch_size,
           "max_input_len": max_seq_len,
           "max_kv_cache_capacity": max_seq_len,
           "context_emb": context_hidden_size is not None,
           ...
       },
   }
   # optional VLA fields
   edge_config["context_hidden_size"] = ...   # -> contextEmbDim
   edge_config["image_token_id"] = ...

Step 3 — Runtime parses config.json into EngineConfig
-----------------------------------------------------

``LLMEngineRunner`` reads those fields into ``mEngineConfig``:

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - config.json field
     - EngineConfig field
     - Drives
   * - ``builder_config.max_batch_size``
     - ``maxSupportedBatchSize``
     - Batch dim of every activation buffer.
   * - ``builder_config.max_input_len``
     - ``maxSupportedInputLength``
     - Sequence dim of ``inputs_embeds`` / context buffers.
   * - ``hidden_size``
     - ``hiddenSize``
     - Width of ``inputs_embeds`` and ``lm_hidden_states``.
   * - ``builder_config.context_emb``
     - ``enableContextEmb``
     - Whether ``context_embs`` output buffer is allocated.
   * - ``context_hidden_size``
     - ``contextEmbDim``
     - Width of the ``context_embs`` buffer.
   * - ``image_token_id``
     - ``imageTokenId``
     - Where vision rows are spliced into ``inputs_embeds``.
   * - engine has ``lm_hidden_states`` out
     - ``enableLmHiddenStates``
     - Whether ``lm_hidden_states`` is exported for the action-context engine.
   * - ``prefix_k`` output shape
     - ``enablePrefixKVOutputs`` + ``prefixKVOutputShape``
     - Whether/what size ``prefix_k`` / ``prefix_v`` buffers are.

Step 4 — Runtime allocates GPU tensors from EngineConfig
--------------------------------------------------------

In the ``LLMInferenceRuntime`` constructor the numbers above become concrete
device allocations:

.. code-block:: cpp

   mInputIds = rt::Tensor({maxSupportedBatchSize, maxSupportedInputLength},
                          kGPU, kINT32);
   mInputsEmbeds = rt::Tensor({maxSupportedBatchSize, maxSupportedInputLength, hiddenSize},
                              kGPU, kHALF);
   mOutputLogits = rt::Tensor({maxSupportedBatchSize, outputVocabSize},
                              kGPU, kFLOAT);

   if (enableContextEmb || enableLmHiddenStates) {
       int32_t seqOutputDim = enableContextEmb ? contextEmbDim : hiddenSize;
       mOutputContextEmbeds = rt::Tensor({maxSupportedBatchSize, maxSupportedInputLength, seqOutputDim},
                                         kGPU, kHALF);
   }
   if (enablePrefixKVOutputs) {
       mOutputPrefixK = rt::Tensor(prefixKVOutputShape, kGPU, kHALF);
       mOutputPrefixV = rt::Tensor(prefixKVOutputShape, kGPU, kHALF);
   }

Note there is **no PyTorch model** here — the buffer shapes come purely from the
exported ``config.json``. This is why an export that writes the wrong
``max_input_len`` or forgets ``context_hidden_size`` produces either an
undersized buffer or a load-time validation error.

Step 5 — Runtime binds buffers to engine tensors by name
--------------------------------------------------------

At execution the runners attach the allocated device pointers to the engine's
named bindings with ``setInputShape`` + ``setTensorAddress``:

.. code-block:: cpp

   context->setInputShape(name.c_str(), tensor.getShape().getTRTDims());
   context->setTensorAddress(name.c_str(), tensor.rawPointer());

For example the action runner wires ``kvcache_start_index``, ``noise_trajectory``,
``time_steps_t0/t1``, per-layer ``k_cache_{i}`` / ``v_cache_{i}``, and writes back
``denoised_trajectory`` — each looked up by the exact binding name string. The
context handoff is a name-keyed copy too:

.. code-block:: cpp

   mActionRunner->copyInputFrom("context_embs", mActionContextRunner->getVlEmbs(), stream);

Because the copy targets the binding named ``context_embs``, the action engine
must have been exported with that input name — which is exactly what
``GROOT_EDGE_IO.action.input_names`` and ``groot/export/diffusion.py`` declare.

The load-time safety check
--------------------------

When the language engine loads, ``LLMEngineRunner`` validates that the config
numbers match the actual engine tensor shapes, e.g.:

.. code-block:: cpp

   Dims contextEmbedsDim = mEngine->getTensorShape(binding_names::kOutputContextEmbeds);
   validate_eq_engine_with_config(mConfig.contextEmbDim, contextEmbedsDim.d[2], "contextEmbDim");

If the engine exports ``prefix_k`` / ``prefix_v`` but the config is missing the
prefix-KV output metadata, load fails outright. This is the guardrail that keeps
the Python export contract and the C++ runtime bindings in sync.

Summary
-------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       EM["Export module<br/>io_spec input/output names"]
       CFG["config.json<br/>names + shapes + flags"]
       ENG["TRT engine<br/>named bindings"]
       EC["EngineConfig<br/>parsed numbers"]
       ALLOC["GPU tensors<br/>mInputsEmbeds, mOutputContextEmbeds, ..."]
       BIND["setInputShape + setTensorAddress<br/>by binding name"]

       EM --> CFG
       EM --> ENG
       CFG --> EC --> ALLOC --> BIND
       ENG --> BIND

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class CFG,EC,ALLOC,BIND nvNode
       class EM,ENG greyNode
