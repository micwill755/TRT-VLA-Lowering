GR00T Action Context Export Example
=====================================

This page walks through the **action_context stage**: projecting language hidden
states into action-head context embeddings, compiled as
``action_context/context.engine``.


Typical tensor shapes (GR00T N1.5, Libero sample)
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Tensor
     - Shape
     - Notes
   * - ``lm_hidden_states`` (input)
     - ``[1, S, 2048]`` fp16
     - Upstream language output; ``S`` ~ prompt length after image splice
   * - ``vl_embs`` / ``context_embs`` (output)
     - ``[1, S, 1536]`` fp16
     - Action-head context width (``backbone_embedding_dim``)
   * - Downstream diffusion binding
     - ``context_embs [1, S, 1536]``
     - Fed as ``encoder_hidden_states`` to the DiT action expert


Export module wiring
--------------------

The hook builds :class:`ContextProjectionExportModule` from three Eagle / action-head
submodules:

.. code-block:: python

   context_module = ContextProjectionExportModule(
       ctx.model.backbone.eagle_linear,
       ctx.model.action_head.vlln,
       ctx.model.action_head.vl_self_attention,
   ).eval().to(device=device, dtype=dtype)

Export traces a single-tensor forward:

.. code-block:: python

   engine_path = save_trt_engine_module(
       context_module,
       (lm_hidden,),
       ctx.engine_root / "action_context",
       engine_file="context.engine",
       input_names=["lm_hidden_states"],
       output_names=["vl_embs"],
       extra_config={
           "batch_size": batch_size,
           "max_seq_len": max_seq_len,
           "hidden_size": hidden_size,           # LM dim (2048)
           "context_hidden_size": context_hidden_size,  # action dim (1536)
       },
       trt_settings={**ctx.trt_settings, "offload_module_to_cpu": True},
   )


Putting it together — context projection
----------------------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       IN["lm_hidden_states<br/>[B, S, H_lm]"]

       LIN["eagle_linear<br/>H_lm → H_ctx"]
       VLLN["vlln (RMSNorm)"]
       VLSA["vl_self_attention<br/>self-attn on context seq"]
       OUT["vl_embs / context_embs<br/>[B, S, H_ctx]"]

       IN --> LIN --> VLLN --> VLSA --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class LIN,VLSA nvNode
       class IN,VLLN,OUT greyNode


Step-by-step
~~~~~~~~~~~~

**1. Linear projection**

``eagle_linear`` maps LM hidden size (2048) to the action backbone embedding
dimension (1536).

**2. Normalization**

``vlln`` RMSNorm stabilizes activations before the context self-attention block.

**3. Context self-attention**

``vl_self_attention`` runs self-attention over the full language sequence in
action-head space. This matches the eager path in
``FlowmatchingActionHead`` / ``get_action`` before the DiT denoising loop.

**4. Edge-LLM binding names**

Export uses ``lm_hidden_states`` → ``vl_embs`` to match ``ActionContextRunner`` in
the C++ runtime. Python inference stages refer to the same tensor as
``context_embs``.


Dummy tensor for downstream export
----------------------------------

Action_context export returns **zero-filled** ``context_embs`` with shape
``[B, S, H_ctx]`` so the diffusion stage can trace without running the context
engine forward immediately after compile.


Stage boundary in the full pipeline
-----------------------------------

.. code-block:: text

   language.lm_hidden  →  action_context  →  context_embs  →  action (DiT)

The diffusion :doc:`../diffusion/groot-example` consumes ``vl_embs`` as
``encoder_hidden_states`` for cross-attention inside the DiT.


Parity tips
-----------

1. Compare the **full** :class:`ContextProjectionExportModule`, not individual
   submodules like ``eagle_linear`` alone.
2. Cast the whole module to one dtype (``fp16``) on both eager and TRT sides.
3. Feed the **same** ``lm_hidden`` tensor to eager and TRT when isolating this
   stage.
4. Benchmark parity key: ``action_context:context_embs``.


*Files:* ``trt/modules/export/language.py`` (:class:`ContextProjectionExportModule`),
``trt/executor/models/groot/export/action_context.py``,
``trt/executor/models/groot/inference/action_context.py``
