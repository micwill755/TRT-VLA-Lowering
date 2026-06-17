# TensorRT Edge LLM — Generic VLA Pipeline Architecture

This document describes the **model-agnostic** vision → language → action pipeline built for
TensorRT Edge LLM. The design replaces hard-coded GROOT/PI0.5 tensor names and wiring with
**declarative IO specs** and **config-driven C++ runners**, so new robot policies can be
exported and run through the same `generic_run_inference` binary.

---

## 1. High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         generic_run_inference (C++)                         │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                         │
         ▼                    ▼                         ▼
  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
  │   Visual     │    │    Language      │    │         Action           │
  │   Runner     │    │ Context Runner   │    │   Velocity Runner        │
  │              │    │                  │    │                          │
  │ pixel_values │    │ inputs_embeds    │    │ x_t / actions  (noise)   │
  │      │       │    │ ctx_len          │    │ timestep                 │
  │      ▼       │    │ kv_cache_*       │    │ prefix_k / context_embs ◄┼── wired from LM
  │ visual_embeds│    │      │           │    │ prefix_v / state         │
  │ image_embeds │    │      ▼           │    │ position_ids, mask, …    │
  └──────┬───────┘    │ hidden_states    │    │      │                   │
         │            │ context_embs     │    │      ▼                   │
         │ (optional  │ prefix_k ────────┼───►│  diffusion.engine        │
         │  pack)     │ prefix_v ────────┼───►│  (one denoise step)      │
         └───────────►│ inputs_embeds    │    │      │                   │
                      └──────────────────┘    │      ▼                   │
                                              │ velocity / pred_velocity │
                                              │                          │
                                              │  rollout loop (C++):     │
                                              │  x_t += dt * velocity    │
                                              └──────────────────────────┘
```

**Design principle:** each TensorRT engine owns the **model-specific forward path** inside
the graph. C++ runners own only the **common runtime contract**: buffer allocation, config
parsing, LM→action wiring, denoising rollout, and fixture I/O.

---

## 2. On-Disk Engine Layout

Both GROOT and PI0.5 export the same directory shape:

```
<engine-root>/
├── visual/
│   ├── config.json          # model_type: "vision" (or generic visual)
│   └── visual.engine
├── language/
│   ├── config.json          # model_type: "language"
│   ├── language.engine
│   └── output_names[]       # 1 or more LM outputs
├── action/
│   ├── config.json          # model_type: "action"
│   ├── diffusion.engine     # single denoising step
│   ├── input_names[]
│   ├── output_names[]
│   ├── lm_to_action_slots   # [[lm_slot, action_slot], …]
│   ├── noise_input_name     # "actions" (GROOT) or "x_t" (PI0.5)
│   ├── timestep_schedule    # "discrete_buckets" | "continuous_flow"
│   └── rollout_dt_sign      # +1 (GROOT) or -1 (PI0.5)
└── fixtures/
    └── pid_<pid>/
        ├── *.bin            # raw tensor blobs for C++ smoke / metrics
        └── out/             # optional inference outputs
```

`config.json` per component stores **shapes, dtypes, and semantic names**. TensorRT may rename
bindings to `output0`, `output1`, … at compile time; runners resolve names → bindings by
index with a fallback warning.

---

## 3. Declarative IO Spec (Python)

**File:** `Test/trt/io_spec.py`

```
PipelineIOSpec
├── vision:   ComponentIOSpec(input_names, output_names)
├── language: ComponentIOSpec(input_names, output_names)
├── action:   ComponentIOSpec(input_names, output_names)
└── lm_to_action_slots: [(lm_output_idx, action_input_idx), …]
```

### Slot wiring model

```
Language outputs (by index)          Action inputs (by index)
─────────────────────────            ─────────────────────────
  [0] hidden_states / context_embs     [0] x_t / actions
  [1] prefix_k                         [1] timestep
  [2] prefix_v                         [2] prefix_k  ◄── slot (1,2)
                                       [3] prefix_v  ◄── slot (2,3)
                                       [4] position_ids
                                       [5] attention_mask
```

Python helpers:

| Helper | Role |
|--------|------|
| `wire_lm_outputs_to_action()` | Splices LM outputs into action compile/run inputs (Python TRT path) |
| `action_rollout_extra_config()` | Writes rollout metadata into `action/config.json` for C++ |
| `to_plugin_info()` | Serializes full IO map for plugin / debug metadata |

### GROOT vs PI0.5 IO tables

| Stage | GROOT | PI0.5 |
|-------|-------|-------|
| **Vision in** | `pixel_values` | `pixel_values` |
| **Vision out** | `visual_embeds` | `image_embeds` |
| **Language in** | `inputs_embeds`, `ctx_len`, `kv_cache_*` | same |
| **Language out** | `context_embs` (1 tensor) | `hidden_states`, `prefix_k`, `prefix_v` |
| **Action in** | `actions`, `timestep`, `context_embs`, `state`, `embodiment_id` | `x_t`, `timestep`, `prefix_k`, `prefix_v`, `position_ids`, `attention_mask` |
| **Action out** | `pred_velocity` | `velocity` |
| **LM→action** | `[(0, 2)]` — context → slot 2 | `[(1, 2), (2, 3)]` — KV → slots 2,3 |

### Rollout config

| | GROOT | PI0.5 |
|---|-------|-------|
| `noise_input_name` | `actions` | `x_t` |
| `timestep_schedule` | `discrete_buckets` | `continuous_flow` |
| `rollout_dt_sign` | `+1` | `-1` |
| Timestep dtype | int32 bucket index | float32 in [0, 1] |

---

## 4. Python Export Path

```
pi05_compile_edge_llm.py / groot_compile_edge_llm.py
         │
         ├── save_visual_engine_for_edge_llm()
         │        └── trt/compile.py: save_trt_engine_module()
         │
         ├── save_language_engine_for_edge_llm()
         │        └── Plugin LM wrapper → TRT compile
         │        └── config: language output_names from PipelineIOSpec
         │
         ├── save_action_diffusion_engine_for_edge_llm()
         │        └── StaticActionVelocityStep (one denoise step)
         │        └── extra_config: action_rollout_extra_config() + shapes
         │
         └── _dump_*_edge_fixture()
                  └── trt/compile.py: dump_edge_fixture()
                           └── fixtures/pid_<pid>/*.bin
```

**Fixture dump** (`dump_edge_fixture` in `trt/compile.py`):

- Writes raw contiguous CPU tensor bytes keyed by semantic name.
- One directory per export process: `fixtures/pid_<pid>/`.
- `actions_out.bin` must match the **full padded** action tensor shape (`max_action_dim`),
  not the cropped policy output dim — C++ compares byte-for-byte against `action.getActions()`.

**Model types** stay generic in config: `"vision"`, `"language"`, `"action"`. No
`pi05_language` / `groot_action` suffixes in `model_type`.

---

## 5. C++ Runtime Architecture

### 5.1 Runners

```
┌─────────────────────────┐  ┌──────────────────────────────┐  ┌─────────────────────────────┐
│ GenericVisualEmbedding  │  │ GenericLanguageContextRunner │  │ GenericActionVelocityRunner │
│ Runner                  │  │                              │  │                             │
├─────────────────────────┤  ├──────────────────────────────┤  ├─────────────────────────────┤
│ Reads visual/config     │  │ Reads language/config        │  │ Reads action/config         │
│ Single input/output     │  │ Multi-output support         │  │ Config-driven input names   │
│                         │  │ getOutput(name|index)        │  │ wireLanguageOutputs()       │
│                         │  │ getContextEmbeds() (GROOT)   │  │ isRolloutManagedInput()     │
│                         │  │                              │  │ sampleActionsFromCurrent…() │
│                         │  │                              │  │ inferStep() (--action-step) │
└─────────────────────────┘  └──────────────────────────────┘  └─────────────────────────────┘
```

**Shared context memory:** all three runners can share one TensorRT execution context
allocation (`setSharedContextMemory`) to reduce GPU footprint.

### 5.2 Inference loop (`run_inference.cpp`)

```
FOR each iteration:
  ┌─ optional ─────────────────────────────────────────────┐
  │ visual.infer()  →  visual_embeds                       │
  │ packVisualInputs()  →  language.inputs_embeds        │
  └────────────────────────────────────────────────────────┘

  load inputs_embeds.bin (if not packed)
  language.zeroKvCache()
  language.setContextLength(max_seq_len)
  language.infer()                    ──► LM outputs in GPU buffers

  action.wireLanguageOutputs(language) ──► copy LM outs → action inputs
                                           (fp16→fp32 cast if needed)

  load initial_actions.bin OR initializeNoise(seed)
  action.sampleActionsFromCurrentActions()   ──► full rollout loop
      OR action.inferStep()                  ──► single step (--action-step)

  compare actions_out.bin reference → actionADE / mean_abs
  optionally write outputs to --output-dir
```

### 5.3 LM → action wiring

```
GenericActionVelocityRunner::wireLanguageOutputs()
         │
         │  reads lm_to_action_slots from action/config.json
         │
         ▼
  for (lmSlot, actionSlot) in slots:
      actionName = mInputNames[actionSlot]
      copyInputFrom(actionName, language.getOutput(lmSlot))
         │
         └── copyTensorToInput()
               ├── same dtype  → cudaMemcpy D2D
               └── diff dtype  → D2H → cast (fp16↔fp32) → H2D
                   (LM fp16 KV → action fp32 inputs)
```

Legacy fallback: if `lm_to_action_slots` is empty but action has `context_embs` and LM
exports `context_embs`, copy by name (GROOT backward compat).

### 5.4 Rollout managed vs static inputs

```
Action inputs
├── Rollout-managed (NOT loaded from fixture each run)
│     ├── noise_input_name  (x_t / actions)  ← initial_actions.bin or RNG
│     └── timestep          ← set per denoise step by runner
│
├── LM-wired (each iteration)
│     ├── context_embs  (GROOT)
│     └── prefix_k, prefix_v  (PI0.5)
│
└── Static (loaded once from fixture at startup)
      ├── position_ids.bin
      ├── attention_mask.bin
      ├── state.bin
      └── embodiment_id.bin
```

### 5.5 Denoising schedules

**Discrete buckets (GROOT):**

```
step = 0 .. N-1
  bucket = floor(step / N * num_timestep_buckets)
  timestep[input] = int32(bucket)
  infer engine → pred_velocity
  actions += (+1) * dt * pred_velocity
```

**Continuous flow (PI0.5):**

```
step = 0 .. N-1
  t = 1.0 + step * (-1/N)        # 1.0 → 0.0
  timestep[input] = float32(t)
  infer engine → velocity
  x_t += (-1) * dt * velocity
```

---

## 6. End-to-End Data Flow (PI0.5 Example)

```
Policy batch (Python)
      │
      ▼
┌─────────────┐     pixel_values.bin
│ Vision TRT  │ ──► (skipped at C++ runtime if inputs_embeds pre-packed)
└─────────────┘
      │
      ▼
┌─────────────┐     inputs_embeds.bin, ctx_len
│ Language TRT│ ──► hidden_states.bin  (debug fixture)
│  (prefill)  │     prefix_k.bin ─────────────┐
└─────────────┘     prefix_v.bin ─────────────┤
      │                                      │
      │         wireLanguageOutputs()        │
      ▼                                      ▼
┌─────────────┐     initial_actions.bin     prefix_k/v → action inputs
│ Action TRT  │ ◄── position_ids.bin
│ (1 step)    │ ◄── attention_mask.bin
└─────────────┘
      │  × num_inference_steps rollout
      ▼
  actions_out.bin (reference, full [B, chunk, max_action_dim])
      │
      ▼
  C++ compares → actionADE ≈ 0.0013 (verified)
```

---

## 7. Fixture Contract

| File | Required | Purpose |
|------|----------|---------|
| `inputs_embeds.bin` | Yes (unless `--pack-visual-inputs`) | Language prefill input |
| `pixel_values.bin` | With `--run-visual` | Vision input |
| `initial_actions.bin` | Optional | Fixed noise; else `--noise-seed` |
| `timestep.bin` | With `--action-step` | Single-step mode |
| `position_ids.bin`, `attention_mask.bin` | PI0.5 | Static action inputs |
| `prefix_k.bin`, `prefix_v.bin` | Debug only | Not used when LM wiring is active |
| `velocity.bin` | Debug | Single-step reference |
| `actions_out.bin` | Optional | Rollout reference for ADE metrics |

**Byte size rule:** reference tensors must match the **engine buffer** exactly
(e.g. `actions_out` = 6400 bytes for shape `[1, 50, 32]` fp32).

---

## 8. Files Changed (Summary)

### Python (`Test/`)

| File | Change |
|------|--------|
| `trt/io_spec.py` | **New.** `PipelineIOSpec`, `GROOT_EDGE_IO`, `PI05_EDGE_IO`, rollout helpers |
| `trt/compile.py` | `dump_tensor_bin()`, `dump_edge_fixture()` |
| `trt/language.py` | Dtype-safe prefix LM eager path; generic LM wrappers |
| `trt/serialize.py` | Config-driven input names for serialized engines |
| `pi05_compile_edge_llm.py` | Wired `PI05_EDGE_IO`, fixture dump, generic `model_type` |
| `groot_compile_edge_llm.py` | Wired `GROOT_EDGE_IO`, rollout metadata, shared fixture helper |

### C++ (`gitlab/TensorRT-Edge-LLM/`)

| File | Change |
|------|--------|
| `cpp/generic/genericLanguageContextRunner.{h,cpp}` | Multi-output LM runner; config-driven names |
| `cpp/generic/genericActionVelocityRunner.{h,cpp}` | Config rollout, `wireLanguageOutputs()`, fp16↔fp32 cast |
| `examples/generic/run_inference.cpp` | Generic wiring, static input load, metrics |

---

## 9. Running

```bash
export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so

# Export (Python)
python pi05_compile_edge_llm.py \
  --engine-dir /tmp/pi05_edge_llm \
  --skip-trt --skip-pytorch --seed 42

# Infer (C++)
FIXTURE=/tmp/pi05_edge_llm/fixtures/pid_<pid>
generic_run_inference \
  --engine-root /tmp/pi05_edge_llm \
  --input-dir "$FIXTURE" \
  --output-dir "$FIXTURE/out" \
  --num-iterations 12 --warmup 3
```

PI0.5 does **not** use `--pack-visual-inputs` (language inputs are pre-exported as
`inputs_embeds.bin`).

---

## 10. Adding a New Policy

```
1. Define PipelineIOSpec  (vision / language / action names + lm_to_action_slots)
2. Define ActionRolloutConfig  (noise name, schedule, dt sign)
3. Implement save_*_engine_for_edge_llm() using save_trt_engine_module()
4. Implement _dump_*_edge_fixture() via dump_edge_fixture()
5. No C++ changes needed if:
     - model_type is generic ("language", "action", "vision")
     - rollout schedule is discrete_buckets or continuous_flow
     - LM outputs map cleanly via lm_to_action_slots
```

If a policy needs a new rollout schedule or extra static inputs, extend
`GenericActionVelocityRunner` and `action_rollout_extra_config()` — keep model-specific
logic out of `run_inference.cpp` whenever possible.

---

## 11. Known Constraints

- **Dtype bridge:** LM engines typically output fp16; action engines may expect fp32 for
  KV tensors. C++ casts on wire; ideally export dtypes would align long-term.
- **Binding names:** TRT renames I/O to `output0`, … — cosmetic warnings only.
- **Action dim:** Engine uses padded `max_action_dim`; crop to policy output dim in the
  application layer after `getActions()`.
- **Python serialized path** (`SerializedPI05Action`, plugin inference) is not yet fully
  wired to `io_spec` — C++ generic path is the reference runtime.
