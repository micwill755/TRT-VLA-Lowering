Action Rollout
==============

The compiled **action engine** implements exactly **one** denoising step:

.. code-block:: text

   velocity = f(x_t, timestep, context)

It does not initialize noise, choose timesteps, integrate velocities, or loop.
That outer **rollout** lives in ``trt/executor/models/groot/inference/diffusion.py`` and mirrors the C++
``ActionRunner`` loop at runtime.


Why rollout is separate from the engine
---------------------------------------

Three reasons the loop is not traced into the TRT graph:

1. **Control flow** — ``for step in range(N)``, discrete timestep buckets, and
   per-model schedule math are easier to keep in Python/C++ than inside one fused
   engine.
2. **One engine, many calls** — export compiles a single
   ``StaticActionVelocityStepExportModule`` step; runtime invokes it
   ``num_inference_timesteps`` times.
3. **Backend-agnostic parity** — the rollout takes ``action_runner`` as a
   callable (eager module, in-memory TRT, or serialized engine) so benchmark and
   parity harnesses run **identical** stepping for all backends.


Two layers of generalization
----------------------------

Diffusion export uses the same pattern twice:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Layer
     - Generic piece
     - Model-specific piece
   * - **One step** (inside engine)
     - ``StaticActionVelocityStepExportModule``
     - ``step_encoder``, ``action_expert``, ``velocity_decoder``
   * - **The loop** (outside engine)
     - ``sample_actions_raw``
     - ``ActionRolloutAdapter`` implementation


The loop structure is invariant; five hooks differ per VLA (see below).


sample_actions_raw
------------------

The shared rollout driver:

.. code-block:: python

   @torch.no_grad()
   def sample_actions_raw(action_runner, context, adapter):
       actions = adapter.initial_actions(context)

       for step in range(adapter.num_steps(context)):
           timestep = adapter.make_timestep(step, actions, context)
           runner_inputs = adapter.make_runner_inputs(actions, timestep, context)

           model_output = action_runner(*runner_inputs)
           if isinstance(model_output, (tuple, list)):
               model_output = model_output[0]

           actions = adapter.update(actions, model_output, step, context)

       return adapter.finalize(actions, context)

``action_runner`` is whatever executes one step — eager
``StaticActionVelocityStepExportModule``, a Torch-TRT module, or a deserialized
engine wrapper. The rollout does not branch on backend.


ActionRolloutContext
--------------------

A dataclass holding **inputs that stay constant across all denoise steps** for
one inference pass:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Used by
   * - ``noise``
     - Initial action trajectory (cloned in ``initial_actions``).
   * - ``context_embs``
     - GR00T — ``vl_embs`` from action_context stage.
   * - ``state``, ``embodiment_id``
     - GR00T — proprio and category-specific decoder weights.
   * - ``prefix_k``, ``prefix_v``, ``prefix_pad_mask``
     - Pi0.5, SmolVLA, Alpamayo — language prefix KV cache.
   * - ``encoder_attention_mask``
     - MolmoAct2 — encoder mask for cross-attention.


ActionRolloutAdapter
--------------------

Protocol with five hooks. Each VLA supplies an adapter; the loop never changes.


.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Hook
     - Responsibility
   * - ``initial_actions(context)``
     - Starting ``x_t`` (usually ``context.noise`` with correct dtype/device).
   * - ``num_steps(context)``
     - How many Euler iterations (e.g. ``action_head.num_inference_timesteps``).
   * - ``make_timestep(step, actions, context)``
     - Scalar timestep tensor for this iteration — **model-specific schedule**.
   * - ``make_runner_inputs(actions, timestep, context)``
     - Tuple passed to ``action_runner(*inputs)`` — **model-specific I/O layout**.
   * - ``update(actions, model_output, step, context)``
     - Euler integration: ``actions + dt * velocity``.
   * - ``finalize(actions, context)``
     - Optional post-process (default: return ``actions`` unchanged).


Per-model adapters
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Adapter
     - VLA
     - Distinct behavior
   * - ``GROOTActionAdapter``
     - GR00T
     - Discrete timestep **buckets** (``int(t_cont * num_timestep_buckets)``);
       inputs ``(actions, t, context_embs, state, embodiment_id)``; ``dt = +1/N``.
   * - ``PrefixKVFlowActionAdapter``
     - Pi0.5, SmolVLA
     - Continuous time **counting down** (``1 + step*dt``, ``dt = -1/N``);
       prefix KV via ``make_runner_inputs``.
   * - ``EncoderKVFlowActionAdapter``
     - MolmoAct2
     - ``timestep = step/num_steps``; stacked encoder K/V + attention mask.


Timestep schedule comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GR00T (integer buckets, count up):

.. code-block:: python

   t_cont = step / float(num_steps)
   timestep_bucket = int(t_cont * num_timestep_buckets)
   # e.g. step 0 → bucket 0, step 3 of 4 → bucket 750 (if buckets=1000)

Pi0.5 (continuous, count down):

.. code-block:: python

   dt = -1.0 / num_steps
   timestep = 1.0 + step * dt
   # e.g. N=4: 1.0, 0.75, 0.5, 0.25

Hardcoding one schedule in the rollout would break the other models — hence the
adapter split.


End-to-end flow (GR00T)
------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       CTX["ActionRolloutContext<br/>noise, context_embs, state, embodiment_id"]
       ADP["GROOTActionAdapter"]
       LOOP["sample_actions_raw<br/>for step in 0..N-1"]
       ENG["action_runner<br/>StaticActionVelocityStep / .engine"]
       OUT["final actions [B, horizon, dim]"]

       CTX --> ADP
       ADP --> LOOP
       LOOP -->|"make_runner_inputs"| ENG
       ENG -->|"velocity"| LOOP
       LOOP -->|"update: x += dt*v"| LOOP
       LOOP --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class ADP,ENG nvNode
       class CTX,LOOP,OUT greyNode


Relationship to eager FlowmatchingActionHead
--------------------------------------------

Eager GR00T inlines the loop in ``get_action``:

.. code-block:: python

   actions = torch.randn(...)
   for t in range(num_steps):
       ...
       pred_velocity = pred[:, -self.action_horizon :]
       actions = actions + dt * pred_velocity

Export + rollout split that into:

- **Engine** — one ``StaticActionVelocityStep`` call (the body of the loop body).
- **``GROOTActionAdapter``** — ``make_timestep``, ``make_runner_inputs``, ``update``.
- **``sample_actions_raw``** — the ``for t in range(num_steps)`` wrapper.

For parity, run eager and TRT through ``sample_actions_raw`` with the same
``ActionRolloutContext`` and seed so differences isolate to the step engine, not
the loop.


When you can skip rollout
-------------------------

For a quick one-off GR00T script you can call ``action_head.get_action`` directly
and never touch the groot inference diffusion rollout helper.

Use rollout when you need to:

- drive a **compiled single-step engine** (no loop inside the graph),
- compare **eager vs in-memory vs serialized** with identical stepping,
- support **multiple VLAs** without copying the loop,
- match **C++ ActionRunner** behavior in Python benchmarks.


Related pages
-------------

- :doc:`overview` — why one step is compiled, loop stays outside.
- :doc:`components` — ``StaticActionVelocityStepExportModule`` and step encoders.
- :doc:`groot-example` — tensor shapes for one GR00T denoising step.

*Files:* ``trt/executor/models/groot/inference/diffusion.py``,
``trt/executor/models/groot/export/diffusion.py``
