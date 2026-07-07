Customization Guide
===================

New VLA model enablement should follow the **profile + pipeline + hooks** workflow
in this repository. Export compiles TensorRT engines compatible with
`TensorRT Edge-LLM <https://nvidia.github.io/TensorRT-Edge-LLM/latest/>`_;
inference and benchmark load those engines and run parity checks against eager
PyTorch.


Architecture Overview
---------------------

A single run is driven by :class:`EdgeOrchestrator` and one shared
:class:`EdgeContext`:

.. code-block:: text

   app.py --model <profile>
        │
        ▼
   VLAProfile (policy, model, prepare_compile_inputs)
        │
        ▼
   EdgeContext  ──► ExportPipeline     (compile engines → engine_root/)
        │          BenchmarkPipeline   (eager vs TRT parity)
        │          InferencePipeline   (per-stage eager / compile / load)
        │
        └── stage_results, inference, benchmark

Each **model** defines export and inference pipeline configs as a **linear stage
graph**. Stages are generic; behavior lives in **hooks** under
``trt/executor/models/<model>/``.

**Export stage loop** (``ExportRunner``):

.. code-block:: text

   pipeline preprocess(ctx) → pipeline_inputs
   for stage in stages:
       stage_inputs = merge(pipeline_inputs, upstream stage_results)
       preprocess? → export → save_artifacts? → postprocess?
       ctx.stage_results[stage_id] = stage dict
   pipeline postprocess?(ctx, stage_results)

**Inference stage loop** (``InferenceRunner``):

.. code-block:: text

   pipeline preprocess(ctx) → pipeline_inputs
   for stage in stages:
       stage_inputs = merge(pipeline_inputs, upstream stage_results)
       preprocess? → compile? | load? → execute → postprocess?
       ctx.stage_results[stage_id] = stage dict
   pipeline postprocess?(ctx, stage_results)

Reference implementation: **GR00T** (``Gr00tN1d7``) — four engines:

.. list-table::
   :header-rows: 1
   :widths: 12 18 30 40

   * - Stage
     - ``engine_subdir``
     - Export hooks
     - Inference hooks
   * - 0 Vision
     - ``visual/``
     - ``groot/export/vision.py``
     - ``groot/inference/vision.py``
   * - 1 Language
     - ``language/``
     - ``groot/export/language.py``
     - ``groot/inference/language.py``
   * - 2 Action context
     - ``action_context/``
     - ``groot/export/action_context.py``
     - ``groot/inference/action_context.py``
   * - 3 Action (diffusion step)
     - ``action/``
     - ``groot/export/diffusion.py``
     - ``groot/inference/diffusion.py``


Pipeline Customization Points
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Area
     - Files
     - What to update
   * - Orchestrator
     - ``trt/orchestrator/edge_orchestrator.py``, ``app.py``
     - CLI entry, ``EdgeContext`` construction, export/benchmark/inference sequencing.
   * - Pipeline registry
     - ``trt/config/pipeline_registry.py``
     - ``register_export_pipeline``, ``register_inference_pipeline``; aliases
       (e.g. ``gr00t`` → ``Gr00tN1d7``).
   * - Export topology
     - ``trt/executor/models/<model>/export/pipeline.py``
     - ``PipelineConfig``: ordered ``StageConfig`` list, pipeline hooks,
       stage ``input_sources`` (DAG edges).
   * - Inference topology
     - ``trt/executor/models/<model>/inference/pipeline.py``
     - Same graph as export; stage hooks use ``preprocess``, ``compile``,
       ``load``, ``execute``, ``postprocess``.
   * - Serialized wrappers
     - ``trt/executor/models/<model>/load/serialize.py``
     - ``SerializedGroot*`` callables used by inference ``load`` hooks.
   * - Generic runners
     - ``trt/pipelines/{export,inference,benchmark}.py``,
       ``trt/runner/{export,inference}.py``
     - Usually unchanged; new models plug in via hooks + config.
   * - Benchmark
     - ``trt/pipelines/benchmark.py``, ``trt/measure.py``
     - Runs inference per execution mode; prints stage parity tables.


Profile Customization Points
----------------------------

Profiles own **HuggingFace / LeRobot setup** and **compile sample inputs**.

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Area
     - Files
     - What to update
   * - Profile base
     - ``trt/profile/base.py``
     - Abstract ``VLAProfile``: policy, model, processors, tokenizers,
       ``prepare_compile_inputs``.
   * - Model profile
     - ``trt/profile/<model>.py``
     - ``GrootProfile``, ``Pi05Profile``, etc.: config, ``_init_policy``,
       ``_init_models``, ``_init_tokenizers``, input packing.
   * - Registration
     - ``trt/profile/registry.py``
     - Add ``"<cli_name>": "trt.profile.<module>:<ProfileClass>"``.
   * - Sample data
     - ``trt/data.py``
     - ``load_test_data``, ``prepare_model_inputs``, model-specific packers
       (e.g. ``pack_groot_language_inputs`` in ``trt/packing.py``).
   * - IO contract
     - ``trt/io_spec.py``
     - ``PipelineIOSpec`` / ``GROOT_EDGE_IO``: TRT binding names, slot wiring
       between language → action_context → action.


Stage Hook Customization Points
-------------------------------

Each stage is a **node** with a **runner** and **hooks** (string paths resolved
at runtime via ``trt.hooks.resolve``).

**Export** (stage hooks):

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Hook
     - Typical module
     - Responsibility
   * - ``preprocess``
     - ``export/<stage>.py``
     - Shape stage inputs from merged pipeline + upstream dict; splice upstream
       tensors when needed.
   * - ``export``
     - ``export/<stage>.py``
     - Patch attention plugins if needed, trace subgraph, ``save_trt_engine_module``,
       return ``engine_path`` and downstream ``tensors``.
   * - ``save_artifacts``
     - ``export/<stage>.py``
     - Non-TRT files (e.g. embedding table, tokenizer JSON beside
       ``language.engine``).
   * - ``postprocess``
     - ``export/<stage>.py``
     - Optional stage cleanup or context updates.

**Inference** (stage hooks):

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Hook
     - Typical module
     - Responsibility
   * - ``preprocess``
     - ``inference/<stage>.py``
     - Shape stage inputs; read upstream ``tensors`` from merged dict.
   * - ``compile``
     - ``inference/<stage>.py``
     - On-the-fly TRT compile when ``execution_mode`` is ``IN_MEMORY``.
   * - ``load``
     - ``inference/<stage>.py`` + ``load/serialize.py``
     - Construct serialized engine wrapper when ``execution_mode`` is
       ``SERIALIZED``.
   * - ``execute``
     - ``inference/<stage>.py``
     - Dispatch eager / in-memory TRT / serialized TRT forward internally.
   * - ``postprocess``
     - ``inference/<stage>.py``
     - Copy outputs into ``ctx.inference`` or ``ctx.actions``.


Shared Module Customization Points
----------------------------------

Reusable export wrappers and builders live outside per-model hooks.

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Area
     - Files
     - What to update
   * - Vision export
     - ``trt/modules/export/vision.py``, ``trt/vision.py``
     - ``GridVisionExportModule``, ViT TRT settings, ``vit_visual_edge_config``.
   * - Language export
     - ``trt/modules/export/language.py``, ``trt/language.py``
     - ``CausalLMExportModule``, flat Edge-LLM prefill tensors, RoPE, KV layout.
   * - Action / diffusion export
     - ``trt/modules/export/diffusion.py``, ``trt/action_rollout.py``
     - Static velocity step, ``sample_actions_raw``, rollout adapters.
   * - Compile helpers
     - ``trt/compile.py``, ``trt/hooks/export/plan.py``
     - ``save_trt_engine_module``, ``ExportPlan`` dataclass.
   * - Clone / memory
     - ``trt/utils.py``
     - ``clone_hf_module_for_export`` — disposable compile targets; ``ctx.policy``
       must survive for later stages and eager benchmark.


TensorRT Plugin Customization Points
------------------------------------

Attention and ViT lowering use Edge-LLM TensorRT plugins. For a full walkthrough of
this repo's compile-shim stack (registration, custom ops, converters, patching, and
parity), see :doc:`Plugins overview <../plugins/overview>`.
Upstream reference: `TensorRT Plugins Guide
<https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/plugins-guide.html>`_.

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Area
     - Files
     - What to update
   * - Plugin modules
     - ``trt/plugin/attention.py``, ``trt/plugin/plugin_utils.py``
     - ``PluginAttention`` (decoder), ``ViTPluginAttention`` (SigLIP/ViT).
   * - Patch helpers
     - ``trt/plugin/plugin_utils.py``
     - ``patch_language_attention``, ``patch_vision_attention``,
       ``restore_attention``, ``load_plugins_for_trt``.
   * - Dynamo converter
     - ``trt/plugin/plugin_converter.py``
     - Torch-TensorRT custom op lowering for ``attention_plugin``.
   * - Environment
     - Shell / CI
     - Set ``EDGE_LLM_PLUGIN_SO`` to ``libNvInfer_edgellm_plugin.so`` before
       export or benchmark.


Adding A New VLA Model
----------------------

Follow these steps in order. Use GR00T as the template
(``trt/executor/models/groot/``).

1. **Define the IO contract**

   - Add or reuse a ``PipelineIOSpec`` in ``trt/io_spec.py``.
   - List TRT **input_names** / **output_names** per engine (vision, language,
     action_context if used, action).
   - Define ``lm_to_action_context_slots`` / ``context_to_action_slots`` if
     outputs are spliced by index.

2. **Add a profile**

   - Create ``trt/profile/<model>.py`` subclassing ``VLAProfile``.
   - Set ``name``, ``pipeline_model_type``, ``model_id``, ``engine_dir_default``,
     ``io``.
   - Implement ``_init_policy``, ``_init_models``, ``_init_tokenizers``,
     ``prepare_compile_inputs``.
   - Register in ``trt/profile/registry.py``.

3. **Create the export pipeline**

   - Add ``trt/executor/models/<model>/export/pipeline.py`` with
     ``PipelineConfig(stages=...)``.
   - For each stage, implement ``export/<stage>.py``:

     - ``preprocess(ctx, inputs)`` — shape bindings from merged pipeline + upstream dict
     - ``export(ctx, prepared)`` — trace, compile, return ``engine_path`` and ``tensors``
     - ``save_artifacts`` / ``postprocess`` as needed

   - Add ``export/process.py`` for pipeline-level ``preprocess`` / ``postprocess``.

4. **Create the inference pipeline**

   - Mirror the **same** ``stage_id`` and ``input_sources`` graph in
     ``inference/pipeline.py``.
   - Implement ``preprocess``, ``compile``, ``load``, ``execute``, and
     ``postprocess`` per stage in ``inference/<stage>.py``.
   - Reuse packing logic with export (e.g. language embedding splice).

5. **Add serialized wrappers**

   - ``load/serialize.py``: thin wrappers around ``SerializedTRTEngine`` with
     correct positional args per engine, used by inference ``load`` hooks.

6. **Register pipelines**

   - In ``trt/config/pipeline_registry.py`` ``_register_builtin()`` (or a
     model-specific import):

     .. code-block:: python

        register_export_pipeline("MyModel", MY_PIPELINE)
        register_inference_pipeline("MyModel", MY_INFERENCE_PIPELINE)

   - Ensure ``profile.name`` resolves via registry lookup.

7. **Export and verify**

   .. code-block:: bash

      export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
      python app.py --model mymodel --export-only --engine-dir /tmp/my_model_edge_llm

   - Confirm ``engine_root/<subdir>/*.engine`` and ``config.json`` match
     Edge-LLM expectations.

8. **Benchmark and parity**

   .. code-block:: bash

      python app.py --model mymodel --benchmark-only --engine-dir /tmp/my_model_edge_llm

   - Review per-stage parity tables (eager vs in-memory vs serialized).


Adding A Pipeline Stage
-----------------------

To insert a new engine into an existing model (e.g. a preprocessor between
language and action):

1. Assign the next ``stage_id`` and set ``input_sources`` to upstream stage
   id(s).
2. Add export hooks (``preprocess``, ``export``, ``postprocess``) and matching
   inference hooks (``preprocess``, ``compile``, ``load``, ``execute``,
   ``postprocess``).
3. Add a serialized wrapper in ``load/serialize.py`` if the stage has a TRT
   engine.
4. Append ``StageConfig`` to export and inference ``pipeline.py`` tuples.
5. Update ``PipelineIOSpec`` and downstream stage ``preprocess`` wiring.


Component Reference
-------------------

Typical Edge-LLM VLA layout (GR00T-style):

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Component
     - Output directory
     - Primary TRT outputs
     - Downstream consumer
   * - Vision
     - ``visual/``
     - ``visual_embeds`` ``[N, H]``
     - Language packing (image token slots in ``inputs_embeds``)
   * - Language
     - ``language/``
     - ``logits``, ``lm_hidden_states``, ``prefix_k``, ``prefix_v``
     - Action context (``lm_hidden_states``)
   * - Action context
     - ``action_context/``
     - ``vl_embs`` / ``context_embs``
     - Action diffusion rollout
   * - Action
     - ``action/``
     - ``velocity`` (single denoising step)
     - Python loop: ``sample_actions_raw`` over ``num_inference_timesteps``

PI0.5 / SmolVLA may omit ``action_context`` and wire ``prefix_k`` / ``prefix_v``
directly into the action engine; see ``PI05_EDGE_IO`` in ``trt/io_spec.py``.


StageConfig Fields
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``stage_id``
     - Unique index; referenced by ``input_sources`` and ``ctx.stage_results``.
   * - ``input_sources``
     - Tuple of upstream ``stage_id`` values; ``()`` = graph entry (uses
       ``ctx.model_inputs``).
   * - ``runner``
     - ``trt.runner.export:ExportRunner`` or
       ``trt.runner.inference:InferenceRunner``.
   * - ``hooks``
     - ``StageHooks`` string paths to callables.
   * - ``engine_subdir``
     - Subdirectory under ``engine_root`` (e.g. ``visual``, ``language``).
   * - ``final_output``
     - Inference only: stage whose ``actions`` tensor becomes ``ctx.actions``.


Benchmark Customization Points
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Area
     - Files
     - What to update
   * - Benchmark stages
     - ``trt/executor/benchmark/pipeline.py``
     - ``BenchmarkStageConfig(name, enabled, run)`` per execution mode.
   * - Inference dispatch
     - ``trt/executor/benchmark/run.py``
     - Sets ``ctx.execution_mode``, runs ``InferencePipeline``.
   * - Parity metrics
     - ``trt/measure.py``, ``report_*`` in ``benchmark/run.py``
     - Action ADE, per-tensor ``tensor_parity_metrics``, optional staged checks.


Environment And Prerequisites
-----------------------------

- **GPU** with CUDA and TensorRT / Torch-TensorRT compatible with your PyTorch
  build.
- **``EDGE_LLM_PLUGIN_SO``** — path to ``libNvInfer_edgellm_plugin.so`` (built
  from `TensorRT-Edge-LLM <https://github.com/NVIDIA/TensorRT-Edge-LLM>`_).
- **LeRobot dataset** — default compile sample from ``lerobot/libero`` (override
  with ``--dataset-id``, ``--episode-index``, ``--frame-index``).
- **Policy weights** — ``--model-id`` or profile default (e.g.
  ``nvidia/GR00T-N1.5-3B``).


Checklist Before Landing
------------------------

- [ ] ``PipelineIOSpec`` documents all engine bindings used at export and load.
- [ ] Export and inference graphs share the same ``stage_id`` / ``input_sources``.
- [ ] ``register_*_pipeline`` includes CLI alias used by ``--model``.
- [ ] ``prepare_compile_inputs`` produces the same tensor layout export and
      inference expect.
- [ ] Vision/language compile paths call ``patch_*_attention`` + ``restore_attention``
      on **cloned** modules only.
- [ ] ``save_artifacts`` writes Edge-LLM sidecars (embeddings, tokenizer, chat
      template) required by C++ runtime.
- [ ] ``--benchmark-only`` loads engines and reports action parity within target
      ADE for your model.
- [ ] Staged parity (vision → language → context → action) investigated when
      end-to-end ADE regresses.


Further Reading
---------------

- `TensorRT Edge-LLM Customization Guide
  <https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/customization-guide.html>`_
- `TensorRT Edge-LLM Plugins Guide
  <https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/plugins-guide.html>`_
- `Alpamayo-R1-10B (VLA) example
  <https://nvidia.github.io/TensorRT-Edge-LLM/latest/examples/alpamayo.html>`_ —
  upstream reference for vision + language + action engine layout
