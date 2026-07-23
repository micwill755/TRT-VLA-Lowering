# Alpamayo `test_vla_pipeline` (local)

How to run TensorRT-Edge-LLM’s VLA pytest harness against a local Alpamayo
ONNX/engine tree when the internal CI datasets are not available.

Companion to: [alpamayo_edge_onnx_e2e.md](./alpamayo_edge_onnx_e2e.md)

## What this test is

`TensorRT-Edge-LLM/tests/defs/test_vla_pipeline.py` is a thin pytest wrapper with
three stages:

| Test | What it does |
|------|----------------|
| `test_engine_build` | `llm_build` + `visual_build` + `action_build` |
| `test_e2e_bench` | `action_inference` + minADE / perf checks |
| `test_inference` | `action_inference` + minADE check |

For `ModelType.VLA` the binary is always `action_inference` (not `vla_inference`).

Official CI expects `$EDGELLM_DATA_DIR/updated_datasets/`:

```text
alpamayo_eval_dataset/input.json   # aka test case alpamayo_action_644
alpamayo_action_chat/input.json
```

plus sibling `gt.json` files for minADE. Those datasets are **not** on a typical
dev machine. Alpamayo is also **not** listed in `tests/test_lists/*.yml` yet, so
you must pass `--test-param` explicitly.

## Prerequisites

- Completed smoke path from the e2e runbook (ONNX + engines + fixed
  `processed_chat_template.json`)
- Sample input from `Test/trt/dump_alpamayo_edge_input.py`
- Conda env with `pytest` (on this host: `lerobot`)
- Built binaries under `$EDGE_LLM_ROOT/build/`

```bash
export EDGE_LLM_ROOT=/home/micwilliams/workspace/TensorRT-Edge-LLM
export TEST_ROOT=/home/micwilliams/workspace/Test
export ALPAMAYO_SRC=/home/micwilliams/workspace/alpamayo/src
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
```

## 1. Create a CI-shaped local layout

The harness expects ONNX/engine folder **names**, not the flat export layout.

```bash
CI_ROOT=/tmp/edgellm_ci
ONNX=$CI_ROOT/onnx
ENG=$CI_ROOT/engines
DATA=$CI_ROOT/data
LOGS=$CI_ROOT/logs
WS=$WORKSPACE_DIR/$MODEL_NAME

rm -rf "$CI_ROOT"
mkdir -p "$ONNX/$MODEL_NAME" "$ENG/$MODEL_NAME" \
  "$DATA/updated_datasets/alpamayo_action_chat" "$LOGS"

# ONNX package names used by TestConfig
ln -sfn "$WS/onnx/llm"    "$ONNX/$MODEL_NAME/llm-fp16-fp16"
ln -sfn "$WS/onnx/visual" "$ONNX/$MODEL_NAME/visual-fp16"
ln -sfn "$WS/onnx/action" "$ONNX/$MODEL_NAME/action-fp16"
```

Engine path names depend on the test param (batch / seq / image-token limits).
For the param used below they resolve to:

```text
engines/Alpamayo-R1-10B/llm-FP16-mxil3424-mxbs1-mxlr0/
engines/Alpamayo-R1-10B/visual-fp16-mnit160-mxit18432-mxpiit192/{visual,action}/
```

```bash
LLM_ENG=$ENG/$MODEL_NAME/llm-FP16-mxil3424-mxbs1-mxlr0
VIS_ENG=$ENG/$MODEL_NAME/visual-fp16-mnit160-mxit18432-mxpiit192
mkdir -p "$VIS_ENG"
ln -sfn "$WS/engines/llm"    "$LLM_ENG"
ln -sfn "$WS/engines/visual" "$VIS_ENG/visual"
ln -sfn "$WS/engines/action" "$VIS_ENG/action"
```

## 2. Dump real sample + `gt.json`

Use the dump helper (writes frames, `input_action.json` with request `id`, and
**real** `gt.json` from Physical AI AV egomotion):

```bash
PYTHONPATH=$ALPAMAYO_SRC:$TEST_ROOT \
  /home/micwilliams/miniforge3/envs/lerobot/bin/python \
  $TEST_ROOT/trt/dump_alpamayo_edge_input.py \
  --output-dir $WORKSPACE_DIR/alpamayo_sample \
  --num-traj-samples 6
```

This writes:

```text
$WORKSPACE_DIR/alpamayo_sample/
  frames/frame_00.png ... frame_15.png
  input_action.json   # 6 requests, each with "id": <clip_id>
  gt.json             # clip_id -> gt_xy / ego_history_xyz / ego_history_rot
```

`gt.json` fields match Alpamayo eval / `compute_minade.py`:

| Field | Source |
|-------|--------|
| `gt_xy` | `ego_future_xyz[..., :2]` in t0-local frame (64×2) |
| `ego_history_xyz` | history in t0-local frame (16×3) |
| `ego_history_rot` | history rotations in t0-local frame (16×3×3) |

Install into the harness dataset path:

```bash
DATA=/tmp/edgellm_ci/data
mkdir -p "$DATA/updated_datasets/alpamayo_action_chat"
cp "$WORKSPACE_DIR/alpamayo_sample/input_action.json" \
   "$DATA/updated_datasets/alpamayo_action_chat/input.json"
cp "$WORKSPACE_DIR/alpamayo_sample/gt.json" \
   "$DATA/updated_datasets/alpamayo_action_chat/gt.json"
```

**minADE6 caveat:** Edge `action_inference` has one global `--noiseSeed`. Six
duplicated requests under one seed are identical, so minADE ≡ single-sample ADE.
For true min-of-6, run six inferences with different `--noiseSeed` values and
merge responses that share the same request `id` before scoring.

## 3. Run `test_inference`

```bash
cd $EDGE_LLM_ROOT
export LLM_SDK_DIR=$EDGE_LLM_ROOT
export ONNX_DIR=/tmp/edgellm_ci/onnx
export ENGINE_DIR=/tmp/edgellm_ci/engines
export EDGELLM_DATA_DIR=/tmp/edgellm_ci/data
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
export TEST_LOG_DIR=/tmp/edgellm_ci/logs
export BUILD_DIR=build

PARAM='Alpamayo-R1-10B-fp16-mxsl4096-mxbs1-mxil3424-mnit160-mxit18432-mxpiit192-alpamayo_action_chat'

/home/micwilliams/miniforge3/envs/lerobot/bin/python -m pytest \
  tests/defs/test_vla_pipeline.py::TestVLAPipeline::test_inference \
  --test-param "$PARAM" \
  -v -s --tb=short
```

### Param breakdown

```text
Alpamayo-R1-10B
  fp16
  mxsl4096          # max seq (parsed; KV capacity defaulted for VLA)
  mxbs1             # max batch size
  mxil3424          # max input len (match llm_build)
  mnit160           # min image tokens
  mxit18432         # max image tokens
  mxpiit192         # max image tokens per image
  alpamayo_action_chat   # dataset key -> .../alpamayo_action_chat/input.json
```

Match `mnit` / `mxit` / `mxpiit` / `mxil` / `mxbs` to how you built engines, or
the harness will look under the wrong `ENGINE_DIR` subfolders.

## Success looks like

- Chat template loads from the symlinked `engines/llm` dir
- No `Unknown content type: image`
- `Processing complete: 6/6 batched requests successful`
- Pytest prints `PASSED` only if minADE@6.4s ≤ 0.90 m
  - With real GT + a single `--noiseSeed`, one clip often **fails** this gate
    (e.g. ~2 m ADE). That means scoring is working; it is not yet CI-comparable
    minADE6.
- Outputs under `$TEST_LOG_DIR/`:
  - `Alpamayo-R1-10B-...-alpamayo_action_chat.json`
  - `minade_results.csv` (real GT from Physical AI AV; still single-seed ADE unless you multi-seed)

## Optional: rebuild engines via the harness

If you want `test_engine_build` instead of reusing existing engines:

```bash
/home/micwilliams/miniforge3/envs/lerobot/bin/python -m pytest \
  tests/defs/test_vla_pipeline.py::TestVLAPipeline::test_engine_build \
  --test-param 'Alpamayo-R1-10B-fp16-mxsl4096-mxbs1-mxil3424-mnit160-mxit18432-mxpiit192' \
  -v -s --tb=short
```

That writes into `$ENGINE_DIR/...` (not your original `$WORKSPACE_DIR/.../engines`
tree unless you pointed `ENGINE_DIR` there). After build, re-check sidecars
(`embedding.safetensors`, tokenizer, fixed `processed_chat_template.json`) as in
the e2e runbook.

## Real accuracy (when you have CI data)

```bash
export EDGELLM_DATA_DIR=/path/to/edge_llm_cache   # contains updated_datasets/
# no dummy gt — use the shipped gt.json next to input.json
```

Then use the same pytest command with `alpamayo_action_chat` or
`alpamayo_action_644`. Gate: minADE@6.4s ≤ **0.90 m** (PyTorch ref ~0.82 m on
644 clips).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `LLM_SDK_DIR` / `ONNX_DIR` required | Export env vars from section 3 |
| No tests collected | Pass `--test-param`; Alpamayo is not in YAML lists |
| Engine / tokenizer not found | Symlink paths must match param (section 1) |
| `Unknown content type: image` | Fix `processed_chat_template.json` (e2e runbook) |
| minADE “no usable MINADE_6S” | Need ≥6 responses per clip `id`, plus real `gt.json` from the dump helper |
| Plugin load failure | `export EDGELLM_PLUGIN_PATH=.../libNvInfer_edgellm_plugin.so` |

## Related

- E2E smoke: `Test/docs/edge/alpamayo_edge_onnx_e2e.md`
- Test class: `TensorRT-Edge-LLM/tests/defs/test_vla_pipeline.py`
- minADE script: `TensorRT-Edge-LLM/examples/accuracy/scripts/compute_minade.py`
- Dump helper: `Test/trt/dump_alpamayo_edge_input.py`
