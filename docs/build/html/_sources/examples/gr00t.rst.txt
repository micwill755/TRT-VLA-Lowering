GR00T Export Pipeline Example
=============================

This example shows how GR00T wraps a vision-language-action model as a set of
TensorRT engines and how the export, load, and inference pipelines sit above the
runtime layer. Use it as the template when adding another engine-backed model.

At a high level, the runtime owns the live model, policy, device, sample inputs,
and engine directory. The pipeline layer does not replace that runtime. It
adapts the runtime state into ordered export stages, writes engines, loads those
engines back, and then drives inference through the same stage graph.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       RT["LLMEngineRuntime / orchestrator<br/>model, policy, inputs, device"]
       CTX["EdgeContext<br/>shared run state"]
       EXP["ExportPipeline<br/>compile engines"]
       LOAD["LoadPipeline<br/>deserialize engines"]
       INF["InferencePipeline<br/>run stage graph"]
       BENCH["BenchmarkPipeline<br/>compare backends"]

       RT --> CTX
       CTX --> EXP
       EXP --> CTX
       CTX --> LOAD
       LOAD --> CTX
       CTX --> BENCH
       BENCH --> INF
       INF --> CTX

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class RT,CTX nvNode
       class EXP,LOAD,INF,BENCH greyNode


Design pattern
--------------

GR00T keeps the generic pipeline mechanics separate from model-specific logic:

- ``PipelineConfig`` describes the ordered graph.
- ``StageConfig`` describes one engine boundary.
- ``PipelineHooks`` run once before and after the full stage loop.
- ``StageHooks`` contain model-specific export, glue, inference, metadata, and
  artifact behavior.
- ``ExportRunner`` and ``InferenceRunner`` are generic. They resolve hook paths
  and call them in a fixed order.

For a new model, first decide where the runtime model should be split into
engines. Then write one stage per engine and use hook functions to isolate the
model-specific details.


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

Export starts with a single ``EdgeContext`` built from the runtime. The context
carries ``ctx.model``, ``ctx.policy``, ``ctx.model_inputs``, ``ctx.engine_root``,
and mutable export state. Each export stage reads from that context, optionally
reads upstream ``StageResult`` objects, and writes one new ``StageResult``.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       PRE["pipeline preprocess<br/>normalize runtime inputs"]
       UP["upstream StageResult tensors"]
       GLUE["process_inputs hook<br/>shape current stage inputs"]
       PLAN["plan_export hook<br/>clone subgraph + ExportPlan"]
       COMP["compile hook<br/>torch.export + TRT"]
       SAVE["save_artifacts / metadata hooks"]
       RESULT["StageResult<br/>engine_path, spec, tensors, metadata"]
       ART["ctx.artifacts['stage_N']"]

       PRE --> GLUE
       UP --> GLUE
       GLUE --> PLAN --> COMP --> SAVE --> RESULT --> ART

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class PLAN,COMP,RESULT nvNode
       class PRE,UP,GLUE,SAVE,ART greyNode

The fixed runner order is:

1. Collect upstream results from ``ctx.artifacts`` using ``input_sources``.
2. Start with ``ctx.model_inputs``.
3. Call ``process_inputs`` when the stage needs glue from an upstream engine.
4. Call ``plan_export`` to clone the correct submodule and build an
   ``ExportPlan``.
5. Call ``compile`` to trace and compile the subgraph to a TensorRT engine.
6. Call optional ``metadata``, ``save_artifacts``, and ``after_stage`` hooks.
7. Store ``StageResult`` at ``ctx.artifacts["stage_N"]``.

The important rule is that each stage should export only the subgraph owned by
that engine. Cross-stage reshaping, tensor selection, and bookkeeping belong in
``process_inputs`` glue, not inside the generic runner.


Hook map
--------

GR00T registers dotted hook paths under ``trt.executor.models.groot``. The paths
are resolved at runtime, which keeps the pipeline config declarative.

.. list-table::
   :header-rows: 1
   :widths: 20 32 48

   * - Stage
     - Export hooks
     - Inference hooks
   * - ``visual``
     - ``export.vision:plan_export``, ``export.vision:compile``,
       ``export.vision:metadata``
     - ``inference.vision:run_eager``, ``run_serialized``, ``run_trt``
   * - ``language``
     - ``export.glue:vision_to_language``, ``export.language:plan_export``,
       ``export.language:compile``, ``save_artifacts``, ``metadata``
     - ``inference.glue:vision_to_language``, ``inference.language:run_*``
   * - ``action_context``
     - ``export.glue:language_to_action_context``,
       ``export.action_context:plan_export``, ``compile``, ``metadata``
     - ``inference.glue:language_to_action_context``,
       ``inference.action_context:run_*``
   * - ``action``
     - ``export.glue:action_context_to_action``,
       ``export.action:plan_export``, ``compile``, ``metadata``
     - ``inference.action:run_*``


Inference and engine flow
-------------------------

After export, ``LoadPipeline`` maps each engine directory back to a serialized
handle: ``visual`` -> ``vision``, ``language`` -> ``language``,
``action_context`` -> ``action_context``, and ``action`` -> ``action``. Those
handles are stored on ``ctx.handles.serialized``.

The inference pipeline runs the same stage graph and dispatches each stage by
execution mode:

- ``EAGER`` calls ``run_eager`` on the live runtime model or policy.
- ``SERIALIZED`` calls ``run_serialized`` through loaded engine handles.
- ``IN_MEMORY`` calls ``run_trt`` through in-process TRT modules.

Inference scratch tensors live in ``ctx.inference``. Stage outputs live in
``ctx.stage_results``. When the final ``action`` stage completes, the pipeline
copies ``ctx.stage_results[3].tensors["actions"]`` to ``ctx.actions``.


Edge LLM Runtime
----------------

The C++ entry point in TensorRT Edge-LLM is ``examples/llm/llm_inference.cpp``.
For GR00T, the exported ``language`` stage becomes the main ``--engineDir`` and
the other engines are loaded from ``--multimodalEngineDir``:

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
       loop and copies host actions into ``response.outputActions``.

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
3. Implement ``plan_export`` and ``compile`` for every engine.
4. Implement ``process_inputs`` glue wherever a stage consumes upstream tensors.
5. Add ``save_artifacts`` only for sidecars needed by load or inference.
6. Mirror the same stage IDs in an inference pipeline with
   ``runner="trt.runner.inference:InferenceRunner"``.
7. Add a load pipeline that maps engine subdirectories to serialized handle
   classes.
8. Register export, load, and inference configs in the pipeline registry.

The pipeline should remain a thin wrapper around the runtime: the runtime creates
the model state and sample inputs, while the pipeline declares how that state is
sliced into engines, compiled, loaded, and replayed.

*Files:* ``trt/executor/models/groot/export/pipeline.py``,
``trt/executor/models/groot/inference/pipeline.py``,
``trt/executor/models/groot/load/pipeline.py``, ``trt/runner/export.py``,
``trt/runner/inference.py``, ``trt/context.py``,
``examples/llm/llm_inference.cpp``, ``cpp/runtime/llmInferenceRuntime.cpp``
