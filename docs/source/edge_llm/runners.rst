Runners and the Request Flow
============================

``LLMInferenceRuntime::handleRequest`` runs one request through the runners in a
fixed order. Each runner owns its own TensorRT engine and execution context, but
they all share one execution-context scratch buffer (``mSharedExecContextMemory``,
sized to the max requirement across runners) because they execute serially.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       REQ["request<br/>images + text (+ state)"]
       VR["VisionRunner<br/>visual.engine"]
       SPLICE["embeddingLookup<br/>WithImageInsertion"]
       LR["LLMEngineRunner<br/>language.engine<br/>prefill + decode"]
       ACR["ActionContextRunner<br/>context.engine"]
       AR["ActionRunner<br/>action.engine<br/>denoising loop"]
       RESP["response<br/>outputActions / outputTrajectories"]

       REQ --> VR --> SPLICE --> LR
       LR -->|lm_hidden_states| ACR
       ACR -->|vl_embs| AR
       LR -.->|prefix_k / prefix_v<br/>or context_embs| AR
       AR --> RESP

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class VR,LR,ACR,AR nvNode
       class REQ,SPLICE,RESP greyNode

VisionRunner (VitRunner)
------------------------

When a request carries images, the runtime calls ``mVisionRunner->preprocess(...)``
then ``mVisionRunner->infer(stream)``. The vision engine's ``pixel_values`` binding
is **HWC** ``[batch, H, W, 3]`` — this is why the export step converts the model's
NCHW sample with ``nchw_to_hwc`` before tracing (``trt/vision.py``,
``groot/export/vision.py``). The engine output ``visual_embeds`` is a flat
``[num_image_tokens, hidden_size]`` block fetched via
``mVisionRunner->getOutputEmbedding()``.

The runtime then does an embedding lookup on the text tokens and **splices** the
visual rows into the image-placeholder positions:

.. code-block:: cpp

   kernel::embeddingLookupWithImageInsertion(
       mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(),
       imageEmbedsTensor, mInputsEmbeds, stream);

The placeholder token is identified by ``image_token_id`` from the vision (and
language) ``config.json``. The number of rows each placeholder expands to is the
vision ``seq_len`` written to ``visual/config.json``.

LLMEngineRunner
---------------

The backbone engine loaded from ``engineDir``. At construction the runtime reads
its ``config.json`` into ``mEngineConfig`` and allocates all the large activation
tensors from those numbers (see :doc:`bindings`). For each request:

.. code-block:: cpp

   mLLMEngineRunner->executePrefillStep(
       mInputsEmbeds, mHostContextLengths, deepstackEmbeds,
       mOutputLogits, /*hidden=*/nullopt, stream,
       outputContextEmbeds, outputPrefixK, outputPrefixV);

Which auxiliary outputs are populated is decided entirely by config flags:

- ``enablePrefixKVOutputs`` → engine emits ``prefix_k`` / ``prefix_v``.
- ``enableContextEmb`` → engine emits ``context_embs`` (dim ``contextEmbDim``).
- ``enableLmHiddenStates`` → engine emits ``lm_hidden_states`` (dim ``hiddenSize``).

For pure text generation the runtime then loops ``executeVanillaDecodingStep``
until EOS. For VLA action models the interesting handoff happens after prefill.

ActionRunner and the three handoff modes
----------------------------------------

``ActionRunner`` declares how it consumes language context via
``getContextHandoff()``. The runtime picks exactly one path:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Path
     - Handoff
     - How context reaches the action engine
   * - Prefix-KV trajectory
     - ``PREFIX_KV``
     - Uses the LM's ``LinearKVCache`` directly. ``sampleTrajectory`` runs the
       denoising loop conditioned on the prompt KV cache (Alpamayo-style, with a
       ``<\|traj_future_start\|>`` stop token).
   * - Pi0.5 velocity
     - ``CONTEXT_TENSOR``
     - When the LM exports ``prefix_k`` / ``prefix_v`` and there is **no**
       ``action_context`` engine, the runtime wires those two outputs straight
       into the action engine before ``sampleActions``.
   * - Context tensor
     - ``CONTEXT_TENSOR``
     - When an ``action_context`` engine exists (GR00T), it projects
       ``lm_hidden_states`` → ``vl_embs``, which is copied into the action
       engine's ``context_embs`` input.

The GR00T context path is the most involved:

.. code-block:: cpp

   mActionContextRunner->copyLmHiddenFrom(mOutputContextEmbeds, stream, actualContextLength);
   mActionContextRunner->infer(stream);
   mActionRunner->copyInputFrom("context_embs", mActionContextRunner->getVlEmbs(), stream);
   mActionRunner->sampleActions(stream);
   mActionRunner->copyActionsToHost(response.outputActions, stream);

ActionContextRunner
-------------------

A small bridge engine present **only** for GR00T. It has a single input binding
(``lm_hidden_states``) and a single output binding (``vl_embs``), whose names it
reads from its own ``config.json``. It exists because GR00T's action head needs a
projected context (``eagle_linear → vlln → vl_self_attention``) that is cheaper to
keep as a separate engine than to fold into either the language or action engine.
Pi0.5 and SmolVLA skip it and hand ``prefix_k`` / ``prefix_v`` to the action
engine instead.

Where each runner writes its result
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Output
     - Filled by
   * - ``response.outputTexts`` / ``outputIds``
     - LLM decode loop (text tokens).
   * - ``response.outputTrajectories``
     - ``ActionRunner::sampleTrajectory`` (prefix-KV path).
   * - ``response.outputActions``
     - ``ActionRunner::copyActionsToHost`` (context-tensor / Pi0.5 paths).
