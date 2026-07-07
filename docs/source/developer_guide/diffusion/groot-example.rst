GR00T Diffusion Step Example
============================

This page walks through **one GR00T denoising step** as exported by
:class:`StaticActionVelocityStepExportModule` with :class:`GrootDiTStepEncoderExportModule`.
It matches the parity harness in ``test_vla.py`` and the eager path in
``FlowmatchingActionHead.get_action``.


Typical tensor shapes (GR00T N1.5, Libero sample)
-------------------------------------------------

Shapes below use the common Libero frame-0 prompt (``B=1``). Sequence length
``S`` is dynamic (packed language length after image splice; often ~566).

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Tensor
     - Shape
     - Notes
   * - ``x_t`` (noisy actions)
     - ``[1, 50, 32]``
     - ``action_horizon=50``, ``action_dim=32``
   * - ``timestep``
     - ``[1]`` (int)
     - Discretized flow-matching bucket index
   * - ``vl_embs`` (context)
     - ``[1, S, 1536]``
     - From action_context stage; ``backbone_embedding_dim=1536``
   * - ``state``
     - ``[1, 1, 64]``
     - Packed proprio; ``max_state_dim=64``
   * - ``embodiment_id``
     - ``[1]`` (int)
     - Selects category-specific weights
   * - ``velocity`` (output)
     - ``[1, 50, 32]``
     - One Euler update worth of action-space velocity


Export module wiring
--------------------

.. code-block:: python

   action_module = StaticActionVelocityStepExportModule(
       step_encoder=GrootDiTStepEncoderExportModule(
           model.action_head,
           embodiment_id=embodiment_id,  # enables dynamic TRT wrappers
       ),
       action_expert=model.action_head.model,  # DiT
       velocity_decoder=TRTDynamicCategorySpecificMLPExportModule(
           model.action_head.action_decoder
       ),
       output_tokens=model.action_head.config.action_horizon,
       cast_hidden_fp32=False,
   ).eval().to(device=device, dtype=torch.float16)


Putting it together — one GR00T step
------------------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       IN["Inputs<br/>x_t [B,50,32]<br/>timestep [B]<br/>vl_embs [B,S,1536]<br/>state [B,1,64]<br/>embodiment_id [B]"]

       SE["GrootDiTStepEncoder"]
       ST["state_encoder → state_features"]
       AE["action_encoder → action_features"]
       CAT["cat(state, future_tokens, action) → sa_embs"]

       DIT["action_expert (DiT)<br/>sa_embs + vl_embs + timestep"]
       HID["get_action_hidden<br/>last 50 tokens"]
       DEC["velocity_decoder<br/>CategorySpecificMLP"]
       OUT["velocity [B,50,32]"]

       EULER["Euler (outside engine)<br/>x_t = x_t + dt * velocity"]

       IN --> SE
       SE --> ST
       SE --> AE
       ST --> CAT
       AE --> CAT
       CAT --> DIT
       IN --> DIT
       DIT --> HID
       HID --> DEC
       IN --> DEC
       DEC --> OUT
       OUT --> EULER

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class SE,DIT,DEC nvNode
       class IN,ST,AE,CAT,HID,OUT,EULER greyNode


Step-by-step (same flow as the diagram)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Step encoder — build DiT inputs**

.. code-block:: text

   state_features  = state_encoder(state, embodiment_id)           # [B, 1, 1536]
   action_features = action_encoder(x_t, timestep, embodiment_id)  # [B, 50, 1536]
   (+ optional position embedding on action_features)
   future_tokens   = expand(learned future token emb)            # [B, 32, 1536]
   sa_embs         = cat(state_features, future_tokens, action_features, dim=1)
                   # [B, 1+32+50, 1536] = [B, 83, 1536]

**2. Action expert — run DiT**

.. code-block:: text

   model_output = DiT(
       hidden_states=sa_embs,              # [B, 83, inner_dim]
       encoder_hidden_states=vl_embs,      # [B, S, 1536]
       timestep=timestep,
   )

Cross-attention blocks attend from ``sa_embs`` (query stream) to ``vl_embs``
(context / vision-language backbone features).

**3. Select action hidden states**

.. code-block:: text

   action_hidden = model_output[:, -output_tokens:]   # [B, 50, inner_dim]

``output_tokens`` equals ``action_horizon`` (50).

**4. Velocity decoder**

.. code-block:: text

   velocity = action_decoder(action_hidden, embodiment_id)   # [B, 50, 32]

The dynamic MLP wrapper gathers per-batch embodiment weights with
``index_select`` + ``bmm``.

**5. Euler update (runtime, not in TRT engine)**

.. code-block:: text

   x_t = x_t + dt * velocity

Repeat for ``num_inference_timesteps`` (e.g. 4–16 depending on config).


Stage boundaries in the full GR00T pipeline
-------------------------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       V[vision]
       L[language]
       AC[action_context]
       A[action diffusion step]

       V --> L --> AC --> A

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       class V,L,AC,A nvNode

- **vision** → image embeddings.
- **language** → ``lm_hidden`` / ``context_hidden``, prefix KV.
- **action_context** → ``vl_embs`` fed as ``encoder_hidden_states``.
- **action** → ``StaticActionVelocityStep`` engine; :doc:`action-rollout` loops
  over timesteps via ``sample_actions_raw``.


Parity tips
-----------

When validating action-context or diffusion stages in ``test_vla.py``:

1. **Compare the same computation on both sides.** Eager reference must run the
   **full** wrapper (not a single submodule like ``eagle_linear`` alone).
2. **Feed the same input tensor** to eager and TRT when isolating one stage —
   e.g. use ``trt_out[1]`` for both language-derived and TRT-derived paths when
   debugging action_context, not ``lm_hidden_eager`` vs ``trt_out[1]`` mixed.
3. **Cast the whole module** to one dtype (``fp16``) with ``copy.deepcopy`` +
   ``.to(dtype=...)`` on all submodules — match
   ``make_groot_action_context_module`` in the legacy script.
4. **Action ADE** is the acceptance metric; per-stage ``rel_l2`` on velocities
   should be low before end-to-end ADE ``< 0.01`` is achievable.


*Files:* ``test_vla.py``, ``trt/modules/export/diffusion.py``,
``trt/executor/models/groot/export/action.py``,
``lerobot/.../flow_matching_action_head.py``
