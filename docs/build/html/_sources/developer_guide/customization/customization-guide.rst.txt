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
   EdgeContext  ──► ExportPipeline   (compile engines → engine_root/)
        │          LoadPipeline      (deserialize engines → handles.serialized)
        │          BenchmarkPipeline   (eager vs TRT timing + parity)
        │
        └── artifacts, export_state, inference, benchmark

Each **model** defines three pipeline configs (export, inference, load) as a
**linear or DAG stage graph**. Stages are generic; behavior lives in **hooks**
under ``trt/executor/models/<model>/``.

**Export stage loop** (``ExportRunner``):

.. code-block:: text

   preprocess(ctx)
   for stage in stages:
       process_inputs? (glue)
       plan_export  → ExportPlan (module, sample_inputs, engine_dir, cleanup_modules)
       compile      → torch.export trace + TensorRT engine + config.json
       metadata / save_artifacts?
       ctx.artifacts["stage_N"] = StageResult
   postprocess?(ctx)

**Inference stage loop** (``InferenceRunner``):

.. code-block:: text

   preprocess(ctx)
   for stage in stages:
       process_inputs? (glue)
       run_eager | run_serialized | run_trt  (selected by ExecutionMode)
       ctx.stage_results[N] = InferenceStageResult
   pick final_output stage → ctx.actions
   postprocess?(ctx)

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
     - ``groot/export/action.py``
     - ``groot/inference/action.py``


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
     - CLI entry, ``EdgeContext`` construction, export/load/benchmark sequencing.
   * - Pipeline registry
     - ``trt/config/pipeline_registry.py``
     - ``register_export_pipeline``, ``register_inference_pipeline``,
       ``register_load_pipeline``; aliases (e.g. ``gr00t`` → ``Gr00tN1d7``).
   * - Export topology
     - ``trt/executor/models/<model>/export/pipeline.py``
     - ``PipelineConfig``: ordered ``StageConfig`` list, ``PipelineHooks``,
       stage ``input_sources`` (DAG edges).
   * - Inference topology
     - ``trt/executor/models/<model>/inference/pipeline.py``
     - Same graph as export; ``StageHooks`` use ``run_eager`` /
       ``run_serialized`` / ``run_trt``.
   * - Load topology
     - ``trt/executor/models/<model>/load/pipeline.py``,
       ``load/serialize.py``
     - ``SerializedStageSpec(key, engine_subdir, wrapper_cls)`` per engine.
   * - Generic runners
     - ``trt/pipelines/{export,inference,load,benchmark}.py``,
       ``trt/runner/{export,inference}.py``
     - Usually unchanged; new models plug in via hooks + config.
   * - Benchmark
     - ``trt/executor/benchmark/pipeline.py``,
       ``trt/executor/benchmark/run.py``
     - Benchmark stages (eager, serialized TRT), parity reports.


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

**Export** (``StageHooks``):

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Hook
     - Typical module
     - Responsibility
   * - ``plan_export``
     - ``export/<stage>.py``
     - Clone subgraph, build ``ExportPlan`` (module, ``sample_inputs``,
       ``engine_dir``, ``cleanup_modules``).
   * - ``compile``
     - ``export/<stage>.py``
     - Patch attention plugins if needed, ``save_trt_engine_module``, write
       ``config.json`` sidecars.
   * - ``metadata``
     - ``export/<stage>.py``
     - Shapes and fields stored on ``StageResult.metadata`` for downstream
       stages / load.
   * - ``save_artifacts``
     - ``export/<stage>.py``
     - Non-TRT files (e.g. embedding table, tokenizer JSON beside
       ``language.engine``).
   * - ``process_inputs``
     - ``export/glue.py``
     - Inter-stage tensor wiring; may use dummy tensors for trace.

**Inference** (``StageHooks``):

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Hook
     - Typical module
     - Responsibility
   * - ``run_eager``
     - ``inference/<stage>.py``
     - PyTorch parity path on live ``ctx.model`` / ``ctx.policy``.
   * - ``run_serialized``
     - ``inference/<stage>.py``
     - Call ``ctx.handles.serialized.<key>`` (loaded ``.engine``).
   * - ``run_trt``
     - ``inference/<stage>.py``
     - In-process TRT module (``ctx.handles.in_memory``) when populated.
   * - ``process_inputs``
     - ``inference/glue.py``
     - Copy upstream ``StageResult`` tensors into ``ctx.inference`` scratch.


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

Attention and ViT lowering use Edge-LLM TensorRT plugins. See also the upstream
`TensorRT Plugins Guide
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

     - ``plan_export(ctx, stage_inputs) -> ExportPlan``
     - ``compile(plan) -> Path``
     - ``metadata`` / ``save_artifacts`` as needed

   - Add ``export/glue.py`` for ``process_inputs`` between stages.
   - Add ``export/preprocess.py`` / ``postprocess.py`` if the whole pipeline
     needs bookends.

4. **Create the inference pipeline**

   - Mirror the **same** ``stage_id`` and ``input_sources`` graph in
     ``inference/pipeline.py``.
   - Implement ``run_eager``, ``run_serialized``, ``run_trt`` per stage in
     ``inference/<stage>.py``.
   - Reuse or share packing logic with export (e.g. ``pack_*_language_inputs``).

5. **Create the load pipeline**

   - ``load/pipeline.py``: tuple of ``SerializedStageSpec``.
   - ``load/serialize.py``: thin wrappers around ``SerializedTRTEngine`` with
     correct positional args per engine.

6. **Register pipelines**

   - In ``trt/config/pipeline_registry.py`` ``_register_builtin()`` (or a
     model-specific import):

     .. code-block:: python

        register_export_pipeline("MyModel", MY_PIPELINE, aliases=("mymodel",))
        register_inference_pipeline("MyModel", MY_INFERENCE_PIPELINE, aliases=("mymodel",))
        register_load_pipeline("MyModel", MY_LOAD_PIPELINE, aliases=("mymodel",))

   - Ensure ``profile.pipeline_model_type`` or ``profile.name`` resolves via
     aliases.

7. **Export and verify**

   .. code-block:: bash

      export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
      python app.py --model mymodel --export-only --engine-dir /tmp/my_model_edge_llm

   - Confirm ``engine_root/<subdir>/*.engine`` and ``config.json`` match
     Edge-LLM expectations.

8. **Benchmark and parity**

   .. code-block:: bash

      python app.py --model mymodel --benchmark-only --engine-dir /tmp/my_model_edge_llm

   - Review timing summary, action ADE vs eager, and language logits parity
     (``compare_language_logits`` in ``groot/inference/language.py`` as
     reference).


Adding A Pipeline Stage
-----------------------

To insert a new engine into an existing model (e.g. a preprocessor between
language and action):

1. Assign the next ``stage_id`` and set ``input_sources`` to upstream stage
   id(s).
2. Add export + inference hook modules and glue functions.
3. Add ``SerializedStageSpec`` and wrapper in ``load/``.
4. Append ``StageConfig`` to export and inference ``pipeline.py`` tuples.
5. Update ``PipelineIOSpec`` and any ``plan_export`` metadata consumed by
   downstream glue.


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
     - Unique index; referenced by ``input_sources`` and ``ctx.artifacts``.
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
