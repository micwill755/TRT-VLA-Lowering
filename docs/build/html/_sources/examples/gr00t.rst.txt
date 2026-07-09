GR00T Export Pipeline Example
=============================

This example shows how GR00T wraps a vision-language-action model as a set of
TensorRT engines and how the export, load, and inference pipelines sit above the
runtime layer. Use it as the template when adding another engine-backed model.

Quick start
-----------

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so

   # Export + benchmark (default)
   python app.py --model gr00t --device cuda --engine-dir /tmp/groot_edge_llm

   # Export only
   python app.py --model gr00t --export-only --device cuda --engine-dir /tmp/groot_edge_llm

   # Benchmark only (requires prior export)
   python app.py --model gr00t --benchmark-only --device cuda --engine-dir /tmp/groot_edge_llm

   # Eager inference only
   python app.py --model gr00t --inference-only --device cuda

For per-module TRT compile and timing without the full orchestrator:

.. code-block:: bash

   python test_vla_gr00t_e2e.py

At a high level, the orchestrator owns the live model, policy, device, sample inputs,
and engine directory. The pipeline layer adapts that state into ordered export
stages, writes engines, and drives inference and benchmark through the same
stage graph.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       RT["EdgeOrchestrator / EdgeContext<br/>model, policy, inputs, device"]
       CTX["EdgeContext<br/>shared run state"]
       EXP["ExportPipeline<br/>compile engines"]
       INF["InferencePipeline<br/>run stage graph"]
       BENCH["BenchmarkPipeline<br/>compare backends"]

       RT --> CTX
       CTX --> EXP
       EXP --> CTX
       CTX --> BENCH
       BENCH --> INF
       INF --> CTX

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class RT,CTX nvNode
       class EXP,INF,BENCH greyNode


Design pattern
--------------

GR00T keeps the generic pipeline mechanics separate from model-specific logic:

- ``PipelineConfig`` describes the ordered graph.
- ``StageConfig`` describes one engine boundary.
- Pipeline ``preprocess`` / ``postprocess`` run once around the stage loop.
- Stage hooks contain model-specific export and inference behavior.
- ``ExportRunner`` and ``InferenceRunner`` are generic. They resolve hook paths
  and call them in a fixed order.

Export and inference share the same top-level loop: merge pipeline inputs with
upstream stage outputs, then run the stage runner. Cross-stage tensor wiring
lives in each downstream stage's ``preprocess`` hook (not in a separate glue
module).


GR00T stages
------------

GR00T exports four engines. The export and inference pipelines use matching
``stage_id`` values and matching ``input_sources`` edges so downstream glue can
reason about the same graph in both modes.

.. list-table::
   :header-rows: 1
   :widths: 12 20 24 44

   * - Stage
     - Engine directory
     - Inputs from
     - Responsibility
   * - ``0``
     - ``visual``
     - Runtime inputs
     - Convert camera tensors into visual embeddings.
   * - ``1``
     - ``language``
     - Stage ``0``
     - Combine visual embeddings with tokenized language inputs and produce LM
       hidden state plus prefix cache tensors.
   * - ``2``
     - ``action_context``
     - Stage ``1``
     - Build the action context embeddings consumed by the action head.
   * - ``3``
     - ``action``
     - Stage ``2``
     - Produce robot action tensors. This is the ``final_output`` stage.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       IN["runtime sample<br/>model_inputs"]
       V["0 visual<br/>visual/"]
       L["1 language<br/>language/"]
       AC["2 action_context<br/>action_context/"]
       A["3 action<br/>action/"]
       OUT["actions"]

       IN --> V --> L --> AC --> A --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class V,L,AC,A nvNode
       class IN,OUT greyNode


Export data flow
----------------

Export starts with a single ``EdgeContext`` built by the orchestrator. Pipeline
``preprocess`` normalizes ``ctx.model_inputs`` into tokenized tensors, pixels,
state, and embodiment id. Each export stage receives a merged input dict:

.. code-block:: text

   stage_inputs = {**pipeline_inputs, **upstream_stage_output}

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       PRE["pipeline preprocess<br/>tokenize + pack state"]
       UP["upstream stage dict<br/>tensors + metadata"]
       SPRE["stage preprocess<br/>shape bindings"]
       EXP["export hook<br/>trace + TRT compile"]
       SAVE["save_artifacts?<br/>language only"]
       POST["stage postprocess"]
       RESULT["ctx.stage_results stage_id"]

       PRE --> SPRE
       UP --> SPRE
       SPRE --> EXP --> SAVE --> POST --> RESULT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class EXP,RESULT nvNode
       class PRE,UP,SPRE,SAVE,POST greyNode

The fixed runner order is:

1. Merge pipeline inputs with upstream ``ctx.stage_results[input_sources[0]]``.
2. Call stage ``preprocess`` to shape tensors for this engine boundary.
3. Call ``export`` to trace, compile, and write the engine under
   ``ctx.engine_root/<engine_subdir>/``.
4. Call ``save_artifacts`` when the stage needs Edge-LLM sidecars (language only).
5. Call stage ``postprocess`` and store the returned dict at
   ``ctx.stage_results[stage_id]``.

Each stage returns representative ``tensors`` for downstream stages (for example
``image_embs`` after vision). Export does not re-run compiled TRT modules after
compile — downstream tensors come from eager forwards or dummy tensors sized to
the engine ABI.


Hook map
--------

GR00T registers dotted hook paths under ``trt.executor.models.groot``. See the
:doc:`../developer_guide/export_modules/overview` pages for how each export
module is wired.

.. list-table::
   :header-rows: 1
   :widths: 20 32 48

   * - Stage
     - Export hooks
     - Inference hooks
   * - ``visual``
     - ``export.vision:preprocess``, ``export``, ``postprocess``
     - ``inference.vision:preprocess``, ``compile``, ``load``, ``execute``,
       ``postprocess``
   * - ``language``
     - ``export.language:preprocess``, ``export``, ``save_artifacts``,
       ``postprocess``
     - ``inference.language:preprocess``, ``compile``, ``load``, ``execute``,
       ``postprocess``
   * - ``action_context``
     - ``export.action_context:preprocess``, ``export``, ``postprocess``
     - ``inference.action_context:preprocess``, ``compile``, ``load``,
       ``execute``, ``postprocess``
   * - ``action``
     - ``export.diffusion:preprocess``, ``export``, ``postprocess``
     - ``inference.diffusion:preprocess``, ``compile``, ``load``, ``execute``,
       ``postprocess``


Inference and engine flow
-------------------------

Serialized engines are loaded per stage by each inference stage's ``load`` hook
when ``ctx.execution_mode`` is ``SERIALIZED``. Wrappers live in
``trt/executor/models/groot/load/serialize.py`` (``SerializedGrootVision``,
``SerializedGrootLanguage``, etc.).

The inference pipeline uses the same stage graph and dispatches each stage by
execution mode inside the ``execute`` hook:

- ``EAGER`` runs live PyTorch (language uses stock HF ``language_model``).
- ``SERIALIZED`` runs through the loaded engine wrapper from the ``load`` hook.
- ``IN_MEMORY`` runs through an on-the-fly TRT module from the ``compile`` hook.

Inference scratch tensors live in ``ctx.inference``. Stage outputs live in
``ctx.stage_results``. The final ``action`` stage writes ``actions`` to
``ctx.actions`` via ``postprocess``.


Edge LLM Runtime
----------------

The C++ entry point in TensorRT Edge-LLM is ``examples/llm/llm_inference.cpp``.
For GR00T, the exported ``language`` stage becomes the main ``--engineDir`` and
the other engines are loaded from ``--multimodalEngineDir``. Step-by-step export
→ inference instructions, tokenization parity (555-token prefix), and smoke-test
paths are in :doc:`../edge_llm/e2e`.

.. code-block:: bash

   llm_inference \
     --engineDir=/tmp/groot_edge_llm/language \
     --multimodalEngineDir=/tmp/groot_edge_llm \
     --inputFile=requests.json \
     --outputFile=responses.json

``llm_inference.cpp`` parses ``--engineDir`` and ``--multimodalEngineDir``, builds
``rt::LLMInferenceRuntime(args.engineDir, args.multimodalEngineDir, ...)``, then
calls ``handleRequest(request, response, stream)``. That call is where the Python
export contract has to line up with Edge-LLM runtime bindings.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       CLI["llm_inference.cpp<br/>--engineDir language<br/>--multimodalEngineDir root"]
       RT["LLMInferenceRuntime"]
       V["mVisionRunner<br/>visual/visual.engine"]
       L["mLLMEngineRunner<br/>language/language.engine"]
       AC["mActionContextRunner<br/>action_context/context.engine"]
       A["mActionRunner<br/>action/action.engine"]
       RESP["LLMGenerationResponse<br/>outputActions"]

       CLI --> RT
       RT --> V
       V --> L
       L --> AC
       AC --> A
       A --> RESP

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class RT,L nvNode
       class CLI,V,AC,A,RESP greyNode

Runtime directory alignment:

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Export output
     - C++ owner
     - Runtime use
   * - ``language/language.engine``
     - ``LLMEngineRunner``
     - Main language prefill engine loaded from ``--engineDir``. The same
       directory also carries ``config.json``, ``embedding.safetensors``, and
       tokenizer artifacts.
   * - ``visual/visual.engine``
     - ``MultimodalRunner`` / Vit runner
     - Loaded from ``--multimodalEngineDir/visual`` and used to replace image
       placeholders with visual embedding rows.
   * - ``action_context/context.engine``
     - ``ActionContextRunner``
     - Loaded from ``--multimodalEngineDir/action_context`` and used when the
       language engine exports ``lm_hidden_states`` instead of final
       ``context_embs``.
   * - ``action/action.engine``
     - ``ActionRunner``
     - Loaded from ``--multimodalEngineDir/action``. Runs the GR00T denoising
       loop and copies host actions into ``response.outputActions``. See
       :doc:`../developer_guide/diffusion/groot-example` for how one denoising step
       is exported as :class:`StaticActionVelocityStepExportModule`.

Runtime tensor shape contract:

.. list-table::
   :header-rows: 1
   :widths: 22 26 24 28

   * - Tensor
     - Export binding / source
     - Runtime shape
     - Notes
   * - ``pixel_values``
     - ``visual`` input
     - ``[N_img, H_img, W_img, 3]`` fp16/fp32
     - Export converts the model's NCHW sample to HWC because the Edge-LLM
       Vit runner binds HWC pixels. For GR00T's default profile, images are
       224 x 224 and ``N_img`` is the number of camera frames in the request.
   * - ``visual_embeds``
     - ``visual`` output
     - ``[N_img * S_img, H_lm]`` fp16
     - ``S_img`` is saved as ``visual/config.json`` ``seq_len``. The language
       runtime expands each image placeholder to this many embedding rows.
   * - ``inputs_embeds``
     - ``language`` input
     - ``[B, T_exp, H_lm]`` fp16 on GPU
     - ``LLMInferenceRuntime`` allocates this as
       ``[maxSupportedBatchSize, maxSupportedInputLength, hiddenSize]`` and
       reshapes per request. ``T_exp`` is text length after image-token
       expansion.
   * - ``context_lengths``
     - ``language`` input
     - ``[B]`` int32 on CPU
     - The C++ runner validates this beside ``inputs_embeds`` before prefill.
   * - ``kvcache_start_index``
     - ``language`` input
     - ``[0]`` or ``[B]`` int32
     - Empty for fresh prefill; populated when the runtime reuses KV cache.
   * - ``last_token_ids``
     - ``language`` input
     - ``[B, 1]`` int64
     - Selects the token position used for logits.
   * - ``past_key_values_i``
     - ``language`` input per layer
     - ``[B, 2, KvH, T_exp, D_head]`` fp16
     - One binding per decoder layer. The second axis is key/value.
   * - ``logits``
     - ``language`` output
     - ``[B, V]`` fp32
     - Used for normal text generation. GR00T action export does not depend on
       logits for the split action-context path.
   * - ``lm_hidden_states``
     - ``language`` output
     - ``[B, T_exp, H_lm]`` fp16
     - GR00T enables this auxiliary output so ``ActionContextRunner`` can
       project the LM hidden sequence into action context embeddings.
   * - ``vl_embs``
     - ``action_context`` output
     - ``[B, T_ctx, H_ctx]`` fp16
     - ``T_ctx`` usually matches ``T_exp``. ``H_ctx`` is the GR00T action
       context hidden size saved in ``action_context/config.json``.
   * - ``actions``
     - ``action`` input
     - ``[B_action, T_act, D_act]`` fp16
     - Noisy action trajectory updated by the runtime denoising loop.
   * - ``timestep``
     - ``action`` input
     - ``[B_action]`` int64
     - GR00T uses discrete timestep buckets configured in ``action/config.json``.
   * - ``context_embs``
     - ``action`` input
     - ``[B, T_ctx, H_ctx]`` fp16
     - Copied from ``ActionContextRunner::getVlEmbs()`` into the action runner.
   * - ``state``
     - ``action`` input
     - ``[B_action, 1, D_state]`` fp16
     - Packed proprioceptive state. The default GR00T profile pads to
       ``D_state = 64``.
   * - ``embodiment_id``
     - ``action`` input
     - ``[B_action]`` int64
     - Selects the embodiment-conditioned action decoder branch.
   * - ``velocity``
     - ``action`` output
     - ``[B_action, T_act, D_act]`` fp16
     - One denoising-step velocity. The runtime loops over timesteps and writes
       final actions to ``response.outputActions``.

The main design constraint is that export must produce the exact binding names
and shape metadata that ``LLMInferenceRuntime`` expects. The Python side controls
this through ``GROOT_EDGE_IO``, ``language_edge_llm_config()``,
``vit_visual_edge_config()``, and the action ``extra_config``. The C++ side then
uses those configs to allocate tensors, reshape them for each request, and wire
``lm_hidden_states -> vl_embs -> context_embs -> velocity``.


Tensor transforms per runner
----------------------------

Each runner takes one set of input tensors and produces the tensors that feed
the next runner. The diagram below traces one request through all four engines,
showing the shape at every input and output boundary.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       REQ["request<br/>images + text + state"]

       subgraph VIT ["VisionRunner (visual.engine)"]
           direction TB
           VIN["in: pixel_values<br/>[N_img, H, W, 3]"]
           VOUT["out: visual_embeds<br/>[N_img * S_img, H_lm]"]
           VIN --> VOUT
       end

       SPLICE["embedding lookup + splice<br/>text embeds + visual rows<br/>-> inputs_embeds<br/>[B, T_exp, H_lm]"]

       subgraph LLM ["LLMEngineRunner (language.engine)"]
           direction TB
           LIN["in: inputs_embeds [B, T_exp, H_lm]<br/>context_lengths [B]<br/>rope_rotary_cos_sin<br/>past_key_values_i [B,2,KvH,T_exp,D]"]
           LOUT["out: logits [B, V]<br/>lm_hidden_states [B, T_exp, H_lm]"]
           LIN --> LOUT
       end

       subgraph ACR ["ActionContextRunner (context.engine)"]
           direction TB
           ACIN["in: lm_hidden_states<br/>[B, T_exp, H_lm]"]
           ACOUT["out: vl_embs<br/>[B, T_ctx, H_ctx]"]
           ACIN --> ACOUT
       end

       subgraph ACT ["ActionRunner (action.engine)<br/>denoising loop"]
           direction TB
           AIN["in: actions [B_a, T_act, D_act]<br/>timestep [B_a]<br/>context_embs [B, T_ctx, H_ctx]<br/>state [B_a, 1, D_state]<br/>embodiment_id [B_a]"]
           AOUT["out: velocity<br/>[B_a, T_act, D_act]"]
           AIN --> AOUT
           AOUT -->|update actions,<br/>next timestep| AIN
       end

       RESP["response<br/>outputActions<br/>[B_a, T_act, D_act]"]

       REQ --> VIN
       VOUT --> SPLICE --> LIN
       LOUT -->|lm_hidden_states| ACIN
       ACOUT -->|vl_embs -> context_embs| AIN
       AOUT -->|final step| RESP

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       classDef ioNode fill:#ffffff,stroke:#999,stroke-width:1px,color:#333
       classDef clusterLabel fill:none,stroke:#aaa,stroke-width:1.5px

       class REQ,RESP greyNode
       class SPLICE nvNode
       class VIN,VOUT,LIN,LOUT,ACIN,ACOUT,AIN,AOUT ioNode
       class VIT,LLM,ACR,ACT clusterLabel

Reading the diagram:

- **VisionRunner** turns raw camera pixels into a flat block of LM-width
  embedding rows. The row count ``N_img * S_img`` is fixed by
  ``visual/config.json`` ``seq_len``.
- The runtime does an **embedding lookup** on the text tokens and splices the
  visual rows into the image-placeholder positions, producing
  ``inputs_embeds`` at the expanded prompt length ``T_exp``.
- **LLMEngineRunner** prefills the decoder. For GR00T it emits
  ``lm_hidden_states`` (the auxiliary output) in addition to ``logits``.
- **ActionContextRunner** projects ``lm_hidden_states`` into ``vl_embs``, the
  action-context embeddings, which the runtime copies into the action engine's
  ``context_embs`` binding.
- **ActionRunner** runs the flow-matching loop: each call takes the current
  ``actions`` and ``timestep`` and returns a ``velocity`` step. The runtime
  updates ``actions`` and advances the timestep until the schedule finishes,
  then copies the result into ``response.outputActions``.


Adding another engine-backed model
----------------------------------

Use GR00T as the reference shape:

1. Choose stable engine boundaries and assign one ``stage_id`` per engine.
2. Create export stage configs with ``runner="trt.runner.export:ExportRunner"``.
3. Implement ``preprocess``, ``export``, and ``postprocess`` for every engine.
4. Add ``save_artifacts`` only for sidecars needed by the C++ runtime (language).
5. Mirror the same stage IDs in an inference pipeline with
   ``runner="trt.runner.inference:InferenceRunner"``.
6. Implement ``preprocess``, ``compile``, ``load``, ``execute``, and
   ``postprocess`` per inference stage.
7. Add serialized wrappers under ``load/serialize.py`` for ``SERIALIZED`` mode.
8. Register export and inference configs in ``trt/config/pipeline_registry.py``.

The pipeline should remain a thin wrapper around the runtime: the orchestrator
creates the model state and sample inputs, while the pipeline declares how that
state is sliced into engines, compiled, loaded per stage, and replayed.

*Files:* ``trt/executor/models/groot/export/pipeline.py``,
``trt/executor/models/groot/inference/pipeline.py``,
``trt/executor/models/groot/load/serialize.py``, ``trt/runner/export.py``,
``trt/runner/inference.py``, ``trt/context.py``,
``examples/llm/llm_inference.cpp``, ``cpp/runtime/llmInferenceRuntime.cpp``
