Diffusion Overview
==================

VLA policies that predict actions with **flow matching** or diffusion share the
same outer pattern: start from noise, run many denoising steps, integrate
velocities with Euler updates, and return a final action trajectory.

Torch-TRT pipelines compiles **one denoising step** as a TensorRT engine. The
multi-step loop (noise init, ``dt``, ``actions = actions + dt * velocity``) stays
in the rollout driver — Python ``ActionRolloutContext`` during export validation
and C++ ``ActionRunner`` at runtime.

The code lives under ``trt/modules/export/diffusion.py``.


The design problem
------------------

Each VLA builds denoising inputs differently:

- **GR00T** concatenates ``state | future_tokens | action_features`` and runs a
  **DiT** with ``encoder_hidden_states=vl_embs``.
- **Pi0.5** embeds noisy actions, attaches **prefix KV** from language, and runs
  a **Gemma expert**.
- **SmolVLA** and **Alpamayo** have their own suffix/prefix layouts.

But the **orchestration** is identical every time:

.. code-block:: text

   for each denoise step:
       build inputs from (noisy actions, timestep, context)   # model-specific
       run the action expert (DiT / Gemma / …)               # model-specific module
       pick action-token hidden states                         # mostly generic
       decode hidden → velocity                                # model-specific module
       x = x + dt * velocity                                   # generic (outside engine)

Rather than duplicating that loop per model, export uses one generic wrapper —
:class:`StaticActionVelocityStepExportModule` — and injects three pluggable
pieces per VLA.


Three pluggable roles
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Role
     - Typical GR00T binding
     - Responsibility
   * - ``step_encoder``
     - :class:`GrootDiTStepEncoderExportModule`
     - Turn ``(x_t, timestep, *context)`` into the exact ``action_expert`` and
       ``velocity_decoder`` call signatures.
   * - ``action_expert``
     - ``action_head.model`` (DiT)
     - The denoiser transformer for one step.
   * - ``velocity_decoder``
     - :class:`TRTDynamicCategorySpecificMLPExportModule` around ``action_decoder``
     - Map expert hidden states back to action-space velocity.


What stays outside the compiled step
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Piece
     - Why it is separate
   * - Euler rollout loop
     - Runtime calls the action engine ``num_inference_timesteps`` times; the
       engine only implements ``velocity = f(x_t, t, context)``.
   * - Initial noise sampling
     - Done once before the loop; must match seed across eager/TRT parity runs.
   * - Action-context projection
     - ``eagle_linear → vlln → vl_self_attention`` is its own export stage;
       its output ``vl_embs`` is fed into the diffusion step as context.


High-level export flow
----------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       CTX["context_embs / vl_embs<br/>from action_context stage"]
       NOISE["x_t, timestep<br/>from rollout driver"]
       ENC["step_encoder<br/>model-specific"]
       EXP["action_expert<br/>DiT / Gemma / …"]
       DEC["velocity_decoder<br/>CategorySpecificMLP / …"]
       VEL["velocity<br/>[B, horizon, action_dim]"]
       LOOP["Euler loop outside engine<br/>x = x + dt * velocity"]

       NOISE --> ENC
       CTX --> ENC
       ENC --> EXP
       EXP --> DEC
       DEC --> VEL
       VEL --> LOOP

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class ENC,EXP,DEC nvNode
       class CTX,NOISE,VEL,LOOP greyNode


Related pages
-------------

- :doc:`../export_modules/overview` — all GR00T export stages at a glance.
- :doc:`components` — class-by-class reference for wrappers and encoders.
- :doc:`groot-example` — end-to-end GR00T one-step walkthrough with tensor shapes.
- :doc:`action-rollout` — ``sample_actions_raw``, adapters, and the multi-step loop
  outside the compiled engine.

*Files:* ``trt/modules/export/diffusion.py``,
``trt/executor/models/groot/inference/diffusion.py``,
``trt/executor/models/groot/export/diffusion.py``
