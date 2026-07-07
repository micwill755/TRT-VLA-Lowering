Diffusion Components
====================

This page documents the export wrappers in ``trt/modules/export/diffusion.py``.
They exist so **one** TensorRT engine shape can serve every VLA: only the injected
``step_encoder``, ``action_expert``, and ``velocity_decoder`` change per model.


StaticActionVelocityStepExportModule
------------------------------------

The generic **one denoising step** orchestrator. It is intentionally dumb about
model semantics.

.. code-block:: python

   class StaticActionVelocityStepExportModule(nn.Module):
       def forward(self, x_t, timestep, *inputs):
           expert_args, expert_kwargs, decoder_args, decoder_kwargs = (
               self.step_encoder(x_t, timestep, *inputs)
           )
           expert_out = self.action_expert(*expert_args, **expert_kwargs)
           action_hidden = self.step_encoder.get_action_hidden(
               expert_out, self.output_tokens
           )
           if self.cast_hidden_fp32:
               action_hidden = action_hidden.to(dtype=torch.float32)
           velocity = self.velocity_decoder(
               action_hidden, *decoder_args, **decoder_kwargs
           )
           return self.step_encoder.process_velocity(velocity)

**Constructor arguments**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Meaning
   * - ``step_encoder``
     - Subclass of :class:`ActionStepEncoderExportModule` for this VLA.
   * - ``action_expert``
     - The denoiser module (GR00T DiT, Pi0.5 Gemma expert, etc.).
   * - ``velocity_decoder``
     - Maps expert hidden states to action velocity (often a category-specific MLP).
   * - ``output_tokens``
     - How many trailing sequence positions are action tokens (GR00T:
       ``action_horizon``).
   * - ``cast_hidden_fp32``
     - When ``True``, cast expert hidden to fp32 before the decoder (some models
       need this for numerical stability during TRT compile).


ActionStepEncoderExportModule
-----------------------------

Base **contract** for model-specific step encoding. Subclasses implement
``forward()`` and may override two helpers:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Method
     - Default behavior
   * - ``forward(x_t, timestep, *inputs)``
     - **Must implement.** Returns
       ``(expert_args, expert_kwargs, decoder_args, decoder_kwargs)``.
   * - ``get_action_hidden(expert_out, output_tokens)``
     - Take ``.last_hidden_state`` (or a raw tensor/tuple), keep the last
       ``output_tokens`` positions.
   * - ``process_velocity(velocity)``
     - Identity; override when the decoder output needs reshape/crop (Alpamayo).


``forward()`` return tuple
~~~~~~~~~~~~~~~~~~~~~~~~~~

The four return values fully describe how to call the expert and decoder:

.. code-block:: text

   expert_args, expert_kwargs, decoder_args, decoder_kwargs = step_encoder(...)

- **GR00T** returns empty ``expert_args`` and passes DiT kwargs
  (``hidden_states``, ``encoder_hidden_states``, ``timestep``); decoder gets
  ``(embodiment_id,)``.
- **Pi0.5 / SmolVLA / Alpamayo** return prefix-KV expert kwargs and empty decoder
  side inputs.


Model-specific step encoders
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - VLA / expert style
   * - :class:`GrootDiTStepEncoderExportModule`
     - GR00T — builds ``sa_embs``, calls DiT with ``vl_embs`` cross-attention.
   * - :class:`PrefixKVStepEncoderExportModule`
     - Generic prefix-KV suffix path with an external action embedder.
   * - :class:`PI05PrefixKVStepEncoderExportModule`
     - Pi0.5 — sinusoidal time emb + adaRMS cond on Gemma expert.
   * - :class:`SmolVLAPrefixKVStepEncoderExportModule`
     - SmolVLA — dual-tower expert with ``SmolVLAPrefixPastLayers``.
   * - :class:`AlpamayoPrefixKVStepEncoderExportModule`
     - Alpamayo — reshapes velocity to action-space dims in ``process_velocity``.


GrootDiTStepEncoderExportModule (detail)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mirrors the body of ``FlowmatchingActionHead.get_action`` for **one** Euler step:

1. ``state_encoder(state, embodiment_id)`` → state features.
2. ``action_encoder(x_t, timestep, embodiment_id)`` → noisy-action features.
3. Optional position embedding on action tokens.
4. Concatenate ``state | future_tokens | action_features`` → ``sa_embs``.
5. Package DiT kwargs: ``hidden_states=sa_embs``, ``encoder_hidden_states=vl_embs``,
   ``timestep=timestep``.
6. Pass ``embodiment_id`` through to the decoder as ``decoder_args``.

When ``embodiment_id`` is provided at construction, GR00T category-specific
modules are replaced with TRT-friendly dynamic wrappers (see below).


Velocity decoders and category-specific wrappers
------------------------------------------------

GR00T (and other multi-embodiment policies) store **one weight bank per robot**
in ``CategorySpecificLinear`` / ``CategorySpecificMLP``. At eager runtime,
``embodiment_id`` selects which slice to use.

For TensorRT export, two wrapper styles exist:


Fixed wrappers
~~~~~~~~~~~~~~

:class:`TRTFixedCategorySpecificLinearExportModule` and
:class:`TRTFixedCategorySpecificMLP` bake **one** embodiment's weights at
``__init__``. The ``embodiment_id`` argument remains in signatures for API
compatibility but is ignored. Use when compiling a single robot deployment.


Dynamic wrappers
~~~~~~~~~~~~~~~~

:class:`TRTDynamicCategorySpecificLinearExportModule` and
:class:`TRTDynamicCategorySpecificMLPExportModule` keep the full weight bank and
implement selection with ``index_select`` + ``bmm`` so Torch-TRT can lower the
graph. Use when ``embodiment_id`` must remain a runtime input.


Action encoder wrappers (GR00T)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`TRTGrootActionEncoderExportModule` (fixed embodiment) and
:class:`TRTDynamicGrootActionEncoderExportModule` (dynamic) wrap the three
category-specific linear layers and timestep positional encoding around the
noisy action trajectory — the same math as ``action_head.action_encoder``.


Relationship to eager FlowmatchingActionHead
--------------------------------------------

The eager denoising loop in LeRobot:

.. code-block:: python

   for t in range(num_steps):
       action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
       sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
       model_output = self.model(
           hidden_states=sa_embs,
           encoder_hidden_states=vl_embs,
           timestep=timesteps_tensor,
       )
       pred = self.action_decoder(model_output, embodiment_id)
       pred_velocity = pred[:, -self.action_horizon :]
       actions = actions + dt * pred_velocity

Maps to export as:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Eager (per iteration)
     - Export module
   * - ``action_encoder`` + concat + ``model(...)``
     - ``GrootDiTStepEncoder`` + ``action_expert`` inside
       ``StaticActionVelocityStepExportModule``
   * - ``action_decoder`` + slice last horizon
     - ``velocity_decoder`` + ``get_action_hidden`` / ``output_tokens``
   * - ``actions = actions + dt * pred_velocity``
     - Outside the compiled engine (rollout driver)


*Files:* ``trt/modules/export/diffusion.py``,
``lerobot/.../flow_matching_action_head.py``
