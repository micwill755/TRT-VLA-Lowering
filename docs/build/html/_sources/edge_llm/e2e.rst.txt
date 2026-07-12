End-to-End: Export and ``llm_inference``
========================================

This page walks through the full deployment path that was validated for **GR00T**,
**Pi0.5**, and **SmolVLA**: export engines with ``app.py``, then run the C++
``llm_inference`` binary from TensorRT Edge-LLM. The Python export pipeline and the
C++ runtime are separate programs; they only work together when the exported
``config.json`` files, tokenizer sidecars, and request JSON all match what
``LLMInferenceRuntime`` expects.

Prerequisites
-------------

Build or locate the Edge-LLM TensorRT plugin and point both tools at it:

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so   # Python export
   export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so    # C++ llm_inference

From the project root:

.. code-block:: bash

   cd /path/to/Test

Step 1 — Export engines
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 14 28 58

   * - Model
     - Command
     - Default engine root
   * - GR00T
     - ``python app.py --model gr00t --export-only --device cuda --engine-dir /tmp/groot_edge_llm``
     - ``/tmp/groot_edge_llm``
   * - Pi0.5
     - ``python app.py --model pi05 --export-only --device cuda --engine-dir /tmp/pi05_edge_llm``
     - ``/tmp/pi05_edge_llm``
   * - SmolVLA
     - ``python app.py --model smolvla --export-only --device cuda --engine-dir /tmp/smolvla_edge_llm``
     - ``/tmp/smolvla_edge_llm``

Each export writes a tree like:

.. code-block:: text

   /tmp/<model>_edge_llm/
     visual/visual.engine
       config.json
     language/language.engine
       config.json
       embedding.safetensors
       processed_chat_template.json
       tokenizer assets
     action_context/context.engine     # GR00T only
       config.json
     action/action.engine
       config.json

Step 2 — Run C++ inference
--------------------------

All three models use the same CLI shape. ``--engineDir`` points at the
**language** subdirectory; ``--multimodalEngineDir`` points at the export root:

.. code-block:: bash

   /path/to/llm_inference \
     --engineDir=/tmp/<model>_edge_llm/language \
     --multimodalEngineDir=/tmp/<model>_edge_llm \
     --inputFile=/tmp/<model>_edge_llm/runtime_smoke/input_action.json \
     --outputFile=/tmp/<model>_edge_llm/runtime_smoke/output_e2e.json \
     --maxGenerateLength=0 \
     --dumpOutput \
     --dumpProfile

Set ``max_generate_length`` to ``0`` for action-only VLA runs (prefill + diffusion,
no text decode). Successful responses include an ``actions`` array in the output JSON.

How ``handleRequest`` works
---------------------------

``llm_inference`` constructs ``LLMInferenceRuntime`` from ``--engineDir`` and
``--multimodalEngineDir``, then calls ``handleRequest(request, response, stream)``
once per batched request. Each stage below maps to a ``Stage timings`` log line in
the C++ output.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       REQ["1. Request JSON<br/>images + task text"]
       TOK["2. Tokenize<br/>chat template → input_ids"]
       VIT["3. ViT stage<br/>preprocess + visual.engine<br/>→ visual_embeds"]
       APREP["4. Action preprocess<br/>suffix / state / handoff mode"]
       EMB["5. Embedding assembly<br/>lookup + splice image rows<br/>→ inputs_embeds"]
       PRE["6. LLM Prefill<br/>language.engine<br/>→ prefix_k/v or lm_hidden_states"]
       DEC["7. LLM Generation<br/>decode loop (skipped when<br/>max_generate_length = 0)"]
       AC["8a. Action context<br/>context.engine → vl_embs<br/>(GR00T only)"]
       ACT["8b. Diffusor<br/>action.engine denoise loop<br/>→ outputActions"]
       OUT["9. Response JSON<br/>actions array"]

       REQ --> TOK --> VIT --> APREP --> EMB --> PRE --> DEC
       PRE -->|GR00T| AC --> ACT
       PRE -.->|Pi0.5 / SmolVLA<br/>prefix_k, prefix_v| ACT
       ACT --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class VIT,PRE,AC,ACT nvNode
       class REQ,TOK,APREP,EMB,DEC,OUT greyNode

**Step-by-step summary:**

1. **Parse & tokenize** — messages are formatted with ``processed_chat_template.json``
   and encoded to token IDs; ``<image>`` placeholders are marked for expansion.
2. **ViT** — ``VisionRunner`` preprocesses pixels (HWC), expands image tokens, and
   runs ``visual.engine`` to produce ``visual_embeds``.
3. **Action preprocess** — ``ActionRunner`` prepares static inputs and selects the
   handoff mode (``context_tensor`` for GR00T, ``pi05_prefix_kv`` for Pi0.5/SmolVLA).
4. **Embedding assembly** — ``embeddingLookupWithImageInsertion`` splices vision rows
   into ``inputs_embeds`` at image-placeholder positions.
5. **LLM Prefill** — ``LLMEngineRunner::executePrefillStep`` runs the language engine;
   auxiliary outputs feed the action path (``prefix_k``/``prefix_v`` or
   ``lm_hidden_states``).
6. **LLM Generation** — token-by-token decode until EOS; skipped for VLA action runs
   with ``max_generate_length = 0``.
7. **Diffusor** — GR00T routes ``lm_hidden_states`` through ``action_context`` first;
   Pi0.5 and SmolVLA wire ``prefix_k``/``prefix_v`` directly. ``sampleActions`` runs
   the flow-matching loop for ``num_inference_steps`` and copies actions to host.
8. **Response** — ``outputActions`` is written to the output JSON; ``Stage timings - E2E``
   covers the full request.

See :doc:`runners` for runner ownership details and :doc:`bindings` for how
``config.json`` sizes the buffers each stage binds.

Model layout at runtime
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 14 18 18 18 32

   * - Model
     - Vision
     - Language
     - Action context
     - Action handoff
   * - GR00T
     - VitRunner
     - prefill → ``lm_hidden_states``
     - ``ActionContextRunner`` → ``vl_embs``
     - ``context_embs`` denoising (GR00T buckets)
   * - Pi0.5
     - VitRunner
     - prefill → ``prefix_k`` / ``prefix_v``
     - *(none)*
     - Pi0.5 velocity expert (``pi05_prefix_kv``)
   * - SmolVLA
     - VitRunner
     - prefill → ``prefix_k`` / ``prefix_v``
     - *(none)*
     - SmolVLA expert (same binding layout as Pi0.5)

GR00T is the only model here with a separate ``action_context`` engine. Pi0.5 and
SmolVLA wire language prefix KV directly into the action engine
(``runPi05VelocityAction`` in ``llmInferenceRuntime.cpp``).

Request JSON format
-------------------

``llm_inference`` reads a batch of chat-style requests. For VLA smoke tests, each
request needs images plus task text. GR00T also accepts ``embodiment_id`` on the
batch root.

**GR00T** (two cameras, task string, embodiment id):

.. code-block:: json

   {
     "batch_size": 1,
     "max_generate_length": 0,
     "embodiment_id": 31,
     "requests": [{
       "messages": [{
         "role": "user",
         "content": [
           {"type": "image", "image": "/path/to/camera_0.png"},
           {"type": "image", "image": "/path/to/camera_1.png"},
           {"type": "text", "text": "['put the white mug on the left plate ...']"}
         ]
       }]
     }]
   }

**Pi0.5** (three image slots for the default libero profile — two real cameras plus
one padded empty camera; task + state in text):

.. code-block:: bash

   # Pi0.5 export uses 3 image masks → 3×256 + text tokens = static prefix 830

**SmolVLA** (two 512×512 cameras; task text only in the runtime JSON — state is
embedded in the exported prefix, not passed separately to ``llm_inference`` today):

.. code-block:: json

   {
     "batch_size": 1,
     "max_generate_length": 0,
     "requests": [{
       "messages": [{
         "role": "user",
         "content": [
           {"type": "image", "image": "/path/to/camera_0.png"},
           {"type": "image", "image": "/path/to/camera_1.png"},
           {"type": "text", "text": "put the white mug on the left plate ...\\n"}
         ]
       }]
     }]
   }

The C++ runtime formats prompts using ``processed_chat_template.json``. Image
content items become ``<image>`` placeholders (see ``content_types.image.format``),
which VitRunner expands to ``builder_config.seq_len`` embedding rows per image.

Export contract checklist
-------------------------

These are the fields that **must** be correct for ``llm_inference`` to load and run.
See :doc:`bindings` for the full binding-name reference.

Language (``language/config.json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``builder_config`` with ``max_input_len``, ``max_kv_cache_capacity``, ``max_batch_size``
  — written by :func:`language_edge_llm_config` (``trt/language.py``).
- Static prefill engines use :func:`make_language_edge_input_specs` with
  ``static_prefill_seq_len=True`` so the TRT profile matches the traced prefix length.
- Sidecars in ``language/``: ``embedding.safetensors``, HF tokenizer files,
  ``processed_chat_template.json``.
- Optional outputs drive the action path:
  ``prefix_k`` / ``prefix_v`` (Pi0.5, SmolVLA) or ``lm_hidden_states`` + separate
  ``action_context`` (GR00T).

Vision (``visual/config.json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``model_type`` must be ``vit`` (not ``visual``) so ``MultimodalRunner`` loads
  ``VitRunner``.
- ``builder_config.seq_len`` = **connector output tokens per image** (e.g. 256 for
  GR00T/Pi0.5, 64 for SmolVLA), not the raw SigLIP patch count.
- ``image_token_id`` must match the token that ``<image>`` (or GR00T's
  ``IMG_CONTEXT``) encodes to in the C++ tokenizer.
- Engine input ``pixel_values`` is traced as **HWC** ``[N, H, W, 3]``; export calls
  ``nchw_to_hwc`` before ``save_trt_engine_module``.

Action (``action/config.json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Pi0.5 / SmolVLA: ``lm_to_action_slots`` maps language outputs
  ``[1,2] → [2,3]`` (``prefix_k``, ``prefix_v``).
- ``attention_mask`` must be **float32** for the C++ ``ActionRunner`` suffix builder
  (``preparePi05SuffixInputs`` writes ``0.0`` / large negative mask values). Export
  converts bool masks to float before tracing; the SmolVLA step encoder converts
  back to bool inside the graph.
- ``prefix_seq_len`` must equal the language engine's ``max_input_len``.

Tokenization and prefix length parity
-------------------------------------

The language engine is exported at a **fixed** prefix length. The C++ runtime must
produce exactly that many tokens after VitRunner image expansion, or prefill fails
with a shape mismatch on ``inputs_embeds``.

.. list-table::
   :header-rows: 1
   :widths: 12 14 14 60

   * - Model
     - Export seq
     - Runtime path
     - Parity notes
   * - GR00T
     - 555 (libero default)
     - ``groot_edge_chat_template`` + VitRunner expand
     - Export uses :func:`retokenize_groot_for_edge_llm` so Python matches C++
       (Eagle processor alone produces a longer sequence). Helpers live in
       ``trt/tokenizer.py``.
   * - Pi0.5
     - 830
     - ``pi05_compact_prefix`` template
     - C++ Paligemma tokenizer can differ slightly from Python; smoke prompts may
       need tuning to hit the static length. Three image slots × 256 + text.
   * - SmolVLA
     - 151
     - ``smolvla_compact_prefix`` template
     - Prefix = 2×64 image + language + **state** token. Runtime JSON only supplies
       images + text, so the text portion may need one extra token vs the natural
       task string to match 151.

Expanded image token IDs (``>= vocab_size``) are valid in the C++ path: VitRunner
assigns placeholder IDs starting at ``vocab_size``, and
``embeddingLookupWithImageInsertion`` splices vision rows without a table lookup.
Python export preprocess must treat those positions the same way (mask
``flat_ids >= vocab_size`` when building ``inputs_embeds``).

Common failures
---------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Likely cause / fix
   * - ``Missing required field 'builder_config' in builder_config``
     - Language export missing :func:`language_edge_llm_config`. Re-export with
       current ``export/language.py`` hooks.
   * - ``Static dimension mismatch for inputs_embeds. Set [1,T_run,…], expected [1,T_eng,…]``
     - Tokenization parity: runtime ``T_run`` ≠ exported ``max_input_len``. Align
       chat template, image count, and text (see table above).
   * - ``expanded image token count must match engine output rows``
     - Wrong ``image_token_id`` in ``visual/config.json`` (e.g. SmolVLA needs
       ``<image>`` → 49190, not the fake wrapper token 49189).
   * - ``VitRunner: invalid model type: visual``
     - Set ``model_type: vit`` and nest ``seq_len`` under ``builder_config``.
   * - ``unsupported tensor dtype for host fill: 4`` (bool)
     - Action ``attention_mask`` exported as bool; use float32 mask for Pi0.5-style
       action engines.
   * - ``Failed to wire static action inputs``
     - Often the bool mask issue above, or missing ``prefix_seq_len`` in action config.

Validated smoke paths
---------------------

After a successful export + ``llm_inference`` run:

.. list-table::
   :header-rows: 1
   :widths: 14 36 50

   * - Model
     - Input
     - Output
   * - GR00T
     - ``/tmp/groot_edge_llm/runtime_smoke/input_action.json``
     - ``/tmp/groot_edge_llm/runtime_smoke/output_e2e.json``
   * - Pi0.5
     - ``/tmp/pi05_edge_llm/runtime_smoke/input_action.json``
     - ``/tmp/pi05_edge_llm/runtime_smoke/output_e2e.json``
   * - SmolVLA
     - ``/tmp/smolvla_edge_llm/runtime_smoke/input_action.json``
     - ``/tmp/smolvla_edge_llm/runtime_smoke/output_e2e.json``

Example successful prefill metrics (single libero frame, ``max_generate_length=0``):

.. list-table::
   :header-rows: 1
   :widths: 14 16 16 16 16 22

   * - Model
     - Prefill tokens
     - Images
     - Image tokens
     - Diffusor
     - Notes
   * - GR00T
     - 555
     - 2
     - 512
     - ~113 ms
     - 4 engines, ``action_context`` path
   * - Pi0.5
     - 830
     - 3
     - 768
     - ~29 ms
     - ``prefix_k`` / ``prefix_v`` handoff
   * - SmolVLA
     - 151
     - 2
     - 128
     - ~13 ms
     - 64 tokens/image; state in prefix

Further reading
---------------

- :doc:`overview` — how ``engineDir`` and ``multimodalEngineDir`` map to runners
- :doc:`runners` — ``handleRequest`` order and handoff modes
- :doc:`bindings` — ``config.json`` fields and GPU allocation
- :doc:`../examples/gr00t` — GR00T export pipeline and tensor contract
- :doc:`../examples/pi05` — Pi0.5 three-stage layout
- :doc:`../examples/smolvla` — SmolVLA three-stage layout
