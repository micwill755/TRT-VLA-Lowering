# Compare Alpamayo Torch-TRT and Edge ONNX/TRT

This runbook measures the two working Alpamayo paths on the same machine:

- **Torch-TRT:** `Test/vla/test_vla_alpamayo_e2e.py`
- **Edge:** TensorRT-Edge-LLM ONNX export, engine builds, and `action_inference`

It produces:

1. Per-component and total TensorRT compile times
2. Edge ONNX export time
3. Hot GPU inference timings
4. A generated Markdown comparison report

The default sample is the same Physical AI AV clip used by the Torch-TRT e2e:
`030c760c-ae38-49aa-9ad8-f5650a545d26` at `t0_us=5100000`.

## Scripted run

To run every section below unattended, use the companion script, which writes
the same result bundle to `$RESULTS_DIR`:

```bash
docs/edge/run_alpamayo_compare.sh
```

All paths are overridable via environment variables, and expensive stages can be
skipped when iterating (artifacts are reused):

```bash
SKIP_TORCH=1 SKIP_ONNX=1 SKIP_BUILD=1 SKIP_INFER=1 docs/edge/run_alpamayo_compare.sh
```

The sections below document the same steps for manual/interactive runs.

## Important interpretation

The compile comparison has two useful totals:

- **TRT engine compile:** Torch-TRT `dynamo.compile` total versus the sum of
  Edge `visual_build`, `llm_build`, and `action_build`.
- **Full Edge preparation:** ONNX export plus the three Edge engine builds.

The inference paths are not identical:

- Torch-TRT reports vision, LM **prefill**, and one diffusion step.
- Edge reports vision, LM prefill, LM autoregressive generation, and the full
  action stage (10 diffusion steps).

The generated report therefore compares matching stages directly and reports
Torch's `diffusion step × 10` as an explicitly labeled estimate. Do not compare
Torch's current three-stage total directly with Edge full request latency.

Precision also differs: Edge Alpamayo engines are FP16, while the Torch-TRT LM
uses strong typing and FP32 accumulation for acceptable hidden-state parity.
These results compare the currently validated configurations, not
precision-identical engines.

## 0. Set paths and verify prerequisites

Run every section in the same shell.

```bash
set -euo pipefail

export EDGE_LLM_ROOT=/home/micwilliams/workspace/TensorRT-Edge-LLM
export TEST_ROOT=/home/micwilliams/workspace/Test
export ALPAMAYO_SRC=/home/micwilliams/workspace/alpamayo/src
export LEROBOT_PYTHON=/home/micwilliams/miniforge3/envs/lerobot/bin/python
export EDGELLM_EXPORT=/home/micwilliams/miniforge3/envs/edgellm-export/bin/tensorrt-edgellm-export

export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
export MODEL_DIR=$WORKSPACE_DIR/$MODEL_NAME
export SAMPLE_DIR=$WORKSPACE_DIR/alpamayo_sample
export COMPARE_DIR=$WORKSPACE_DIR/alpamayo_path_compare
export RESULTS_DIR=$COMPARE_DIR/results
export EDGE_LLM_PLUGIN_SO=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_PLUGIN_SO

test -x "$LEROBOT_PYTHON"
test -x "$EDGELLM_EXPORT"
test -f "$EDGE_LLM_PLUGIN_SO"
test -d "$MODEL_DIR"
test -x "$EDGE_LLM_ROOT/build/examples/llm/llm_build"
test -x "$EDGE_LLM_ROOT/build/examples/multimodal/visual_build"
test -x "$EDGE_LLM_ROOT/build/examples/multimodal/action_build"
test -x "$EDGE_LLM_ROOT/build/examples/multimodal/action_inference"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

For cleaner numbers, stop other GPU workloads and keep the GPU power/clock
policy unchanged across both runs.

## 1. Create isolated benchmark directories

This deletes only `$COMPARE_DIR`, not the checkpoint or existing working
engines.

```bash
test "$COMPARE_DIR" = "$WORKSPACE_DIR/alpamayo_path_compare"
rm -rf -- "$COMPARE_DIR"
mkdir -p "$RESULTS_DIR"
```

Record the machine and revisions:

```bash
{
  date --iso-8601=seconds
  uname -a
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  git -C "$TEST_ROOT" rev-parse HEAD
  git -C "$EDGE_LLM_ROOT" rev-parse HEAD
  git -C "$ALPAMAYO_SRC/.." rev-parse HEAD
} | tee "$RESULTS_DIR/environment.txt"
```

## 2. Run and time the Torch-TRT e2e

The script performs fresh `torch_tensorrt.dynamo.compile` calls in each process.
Its compile timers exclude checkpoint loading and `torch.export`.

Expected runtime is several minutes on a 32 GiB GPU.

```bash
cd "$TEST_ROOT"

PYTHONPATH="$ALPAMAYO_SRC:$TEST_ROOT" \
EDGE_LLM_PLUGIN_SO="$EDGE_LLM_PLUGIN_SO" \
"$LEROBOT_PYTHON" vla/test_vla_alpamayo_e2e.py \
  2>&1 | tee "$RESULTS_DIR/torch_trt.log"
```

Confirm that all stages passed and compile metrics were printed:

```bash
rg 'TRT compile|vs C \(TRT\)|trt execute|total speedup' \
  "$RESULTS_DIR/torch_trt.log"
```

## 3. Time a fresh Edge ONNX export

The source checkpoint remains in `$MODEL_DIR`; only the benchmark ONNX output
is new.

```bash
mkdir -p "$COMPARE_DIR/onnx"

/usr/bin/time \
  -f '%e' \
  -o "$RESULTS_DIR/edge_onnx_export_s.txt" \
  env PYTHONPATH="$EDGE_LLM_ROOT" \
  "$EDGELLM_EXPORT" \
    "$MODEL_DIR" \
    "$COMPARE_DIR/onnx" \
    --max-kv-cache-capacity 4096 \
  2>&1 | tee "$RESULTS_DIR/edge_onnx_export.log"

printf 'Edge ONNX export: %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_onnx_export_s.txt")"
```

Expected output:

```bash
test -f "$COMPARE_DIR/onnx/llm/model.onnx"
test -f "$COMPARE_DIR/onnx/visual/model.onnx"
test -f "$COMPARE_DIR/onnx/action/model.onnx"
```

## 4. Time fresh Edge TensorRT engine builds

The C++ executable build is intentionally excluded. These commands measure
model engine compilation only.

`llm_build` treats a missing tokenizer as fatal: after it serializes the engine
it calls `copyTokenizerFiles()`, and if `tokenizer.json` / `tokenizer_config.json`
are not present in the ONNX directory the process exits non-zero even though the
engine was written. The `tensorrt-edgellm-export` step does not emit those two
files, so stage them from the working checkpoint's ONNX tree before building.

```bash
mkdir -p "$COMPARE_DIR/engines/llm"

cp "$MODEL_DIR/onnx/llm/tokenizer.json" "$COMPARE_DIR/onnx/llm/"
cp "$MODEL_DIR/onnx/llm/tokenizer_config.json" "$COMPARE_DIR/onnx/llm/"

cd "$EDGE_LLM_ROOT"

/usr/bin/time \
  -f '%e' \
  -o "$RESULTS_DIR/edge_llm_build_s.txt" \
  ./build/examples/llm/llm_build \
    --onnxDir "$COMPARE_DIR/onnx/llm" \
    --engineDir "$COMPARE_DIR/engines/llm" \
    --maxInputLen 3424 \
    --maxKVCacheCapacity 4096 \
    --maxBatchSize 1 \
  2>&1 | tee "$RESULTS_DIR/edge_llm_build.log"

/usr/bin/time \
  -f '%e' \
  -o "$RESULTS_DIR/edge_visual_build_s.txt" \
  ./build/examples/multimodal/visual_build \
    --onnxDir "$COMPARE_DIR/onnx/visual" \
    --engineDir "$COMPARE_DIR/engines" \
    --minImageTokens 160 \
    --maxImageTokens 18432 \
    --maxImageTokensPerImage 192 \
  2>&1 | tee "$RESULTS_DIR/edge_visual_build.log"

/usr/bin/time \
  -f '%e' \
  -o "$RESULTS_DIR/edge_action_build_s.txt" \
  ./build/examples/multimodal/action_build \
    --onnxDir "$COMPARE_DIR/onnx/action" \
    --engineDir "$COMPARE_DIR/engines" \
    --maxBatchSize 1 \
  2>&1 | tee "$RESULTS_DIR/edge_action_build.log"
```

Confirm the engines:

```bash
test -f "$COMPARE_DIR/engines/llm/llm.engine"
test -f "$COMPARE_DIR/engines/visual/visual.engine"
test -f "$COMPARE_DIR/engines/action/action.engine"

printf 'Edge LLM build:    %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_llm_build_s.txt")"
printf 'Edge visual build: %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_visual_build_s.txt")"
printf 'Edge action build: %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_action_build_s.txt")"
```

## 5. Install runtime sidecars into the benchmark engine tree

Use the already validated sidecars from the working Edge engine tree. This does
not affect compile timing.

```bash
cp "$MODEL_DIR/engines/llm/embedding.safetensors" \
   "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/llm/tokenizer.json" \
   "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/llm/tokenizer_config.json" \
   "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/llm/processed_chat_template.json" \
   "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/visual/preprocessor_config.json" \
   "$COMPARE_DIR/engines/visual/"
```

If one of these files is missing, restore it using
[`alpamayo_edge_onnx_e2e.md`](./alpamayo_edge_onnx_e2e.md) before continuing.

## 6. Dump the shared sample

This can be skipped if `$SAMPLE_DIR/input_action.json` already exists and points
to valid images.

```bash
PYTHONPATH="$ALPAMAYO_SRC:$TEST_ROOT" \
"$LEROBOT_PYTHON" "$TEST_ROOT/trt/dump_alpamayo_edge_input.py" \
  --output-dir "$SAMPLE_DIR"

test -f "$SAMPLE_DIR/input_action.json"
test -f "$SAMPLE_DIR/gt.json"
```

## 7. Create a repeated Edge benchmark input

One process loads the engines once, performs three unprofiled warmups, and then
profiles 10 requests. Repeating the request gives Edge enough samples to report
mean/std/min/max without including engine startup.

```bash
export EDGE_BENCH_REQUESTS=10

"$LEROBOT_PYTHON" - <<'PY'
import copy
import json
import os
from pathlib import Path

sample = Path(os.environ["SAMPLE_DIR"]) / "input_action.json"
output = Path(os.environ["COMPARE_DIR"]) / "edge_benchmark_input.json"
n = int(os.environ["EDGE_BENCH_REQUESTS"])

data = json.loads(sample.read_text())
if not data.get("requests"):
    raise RuntimeError(f"No requests in {sample}")
request = data["requests"][0]
data["batch_size"] = 1
data["requests"] = [copy.deepcopy(request) for _ in range(n)]
for i, item in enumerate(data["requests"]):
    item["id"] = f'{request.get("id", "alpamayo")}-bench-{i:02d}'
output.write_text(json.dumps(data, indent=2) + "\n")
print(f"Wrote {n} requests to {output}")
PY
```

## 8. Run hot Edge inference profiling

`--dumpProfile` enables CUDA stage timers. `--profileOutputFile` writes those
metrics as machine-readable JSON. Warmups are explicitly excluded.

```bash
cd "$EDGE_LLM_ROOT"

./build/examples/multimodal/action_inference \
  --engineDir "$COMPARE_DIR/engines/llm" \
  --multimodalEngineDir "$COMPARE_DIR/engines" \
  --inputFile "$COMPARE_DIR/edge_benchmark_input.json" \
  --outputFile "$COMPARE_DIR/edge_benchmark_output.json" \
  --dumpProfile \
  --profileOutputFile "$RESULTS_DIR/edge_profile.json" \
  --warmup=3 \
  --noiseSeed=42 \
  2>&1 | tee "$RESULTS_DIR/edge_inference.log"
```

Validate the run and inspect stage metrics:

```bash
jq '.responses | length' "$COMPARE_DIR/edge_benchmark_output.json"
jq '.stages[] |
    select(.stage_id == "vision_encoder"
        or .stage_id == "llm_prefill"
        or .stage_id == "llm_generation"
        or .stage_id == "action_inference") |
    {stage_id, total_runs, total_gpu_time_ms, average_time_per_run_ms, gpu_time_stats}' \
  "$RESULTS_DIR/edge_profile.json"
```

## 9. Generate the comparison report

This writes `$RESULTS_DIR/comparison.md`.

```bash
"$LEROBOT_PYTHON" - <<'PY'
import json
import os
import re
from pathlib import Path

results = Path(os.environ["RESULTS_DIR"])
requests = int(os.environ["EDGE_BENCH_REQUESTS"])
torch_log = (results / "torch_trt.log").read_text()
edge_profile = json.loads((results / "edge_profile.json").read_text())

def log_value(label: str) -> float:
    match = re.search(rf"^{re.escape(label)}:\s*([0-9.]+)", torch_log, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Missing {label!r} in torch_trt.log")
    return float(match.group(1))

def seconds_file(name: str) -> float:
    return float((results / name).read_text().strip())

def edge_stage(stage_id: str) -> dict:
    for stage in edge_profile.get("stages", []):
        if stage.get("stage_id") == stage_id:
            return stage
    raise RuntimeError(f"Missing Edge profile stage {stage_id!r}")

torch_compile = {
    "vision": log_value("vision TRT compile"),
    "language": log_value("lm TRT compile"),
    "action": log_value("diffusion TRT compile"),
    "total": log_value("total TRT compile"),
}
edge_compile = {
    "onnx_export": seconds_file("edge_onnx_export_s.txt"),
    "vision": seconds_file("edge_visual_build_s.txt"),
    "language": seconds_file("edge_llm_build_s.txt"),
    "action": seconds_file("edge_action_build_s.txt"),
}
edge_compile["trt_total"] = edge_compile["vision"] + edge_compile["language"] + edge_compile["action"]
edge_compile["full_total"] = edge_compile["onnx_export"] + edge_compile["trt_total"]

torch_infer = {
    "vision": log_value("vision trt execute"),
    "language_prefill": log_value("lm trt execute"),
    "diffusion_step": log_value("diffusion trt execute"),
}
torch_infer["action_10_step_estimate"] = 10.0 * torch_infer["diffusion_step"]
torch_infer["prefill_policy_estimate"] = (
    torch_infer["vision"]
    + torch_infer["language_prefill"]
    + torch_infer["action_10_step_estimate"]
)

edge_ids = ("vision_encoder", "llm_prefill", "llm_generation", "action_inference")
edge_infer = {
    stage_id: float(edge_stage(stage_id)["total_gpu_time_ms"]) / requests
    for stage_id in edge_ids
}
edge_infer["profiled_stage_sum"] = sum(edge_infer.values())

report = f"""# Alpamayo path comparison

Requests profiled by Edge: {requests}; warmups: 3.

## Compilation (wall-clock seconds)

| Component | Torch-TRT `dynamo.compile` | Edge path |
|---|---:|---:|
| Vision | {torch_compile['vision']:.3f} | {edge_compile['vision']:.3f} |
| Language | {torch_compile['language']:.3f} | {edge_compile['language']:.3f} |
| Action / diffusion | {torch_compile['action']:.3f} | {edge_compile['action']:.3f} |
| **TRT total** | **{torch_compile['total']:.3f}** | **{edge_compile['trt_total']:.3f}** |
| ONNX export (creates `.onnx` files) | — | {edge_compile['onnx_export']:.3f} |
| **Full preparation** | — | **{edge_compile['full_total']:.3f}** |

Torch timing excludes `torch.export` and checkpoint loading. Edge TRT total excludes
ONNX export and C++ executable compilation; Edge full preparation includes ONNX
export plus all three TRT engine builds.

## Hot inference (GPU milliseconds per request)

| Stage | Torch-TRT | Edge |
|---|---:|---:|
| Vision | {torch_infer['vision']:.3f} | {edge_infer['vision_encoder']:.3f} |
| LM prefill | {torch_infer['language_prefill']:.3f} | {edge_infer['llm_prefill']:.3f} |
| LM generation | not run | {edge_infer['llm_generation']:.3f} |
| One diffusion step | {torch_infer['diffusion_step']:.3f} | not separately reported |
| Full action stage | ~{torch_infer['action_10_step_estimate']:.3f} (10× estimate) | {edge_infer['action_inference']:.3f} |

Torch estimated vision + prefill + 10 diffusion steps:
**{torch_infer['prefill_policy_estimate']:.3f} ms**.

Edge sum of vision + prefill + generation + action profile stages:
**{edge_infer['profiled_stage_sum']:.3f} ms/request**.

These two totals are not direct equivalents because the Torch e2e does not run
autoregressive CoC generation and its 10-step action figure is an estimate.

## Validity notes

- Torch language uses FP32 accumulation; Edge Alpamayo engines are FP16.
- Torch values are averages of 100 CUDA-event iterations per stage.
- Edge values are total profiled GPU time divided by {requests} requests.
- Engine loading, tokenization, image I/O, and other host overhead are excluded.
"""

path = results / "comparison.md"
path.write_text(report)
print(report)
print(f"Wrote {path}")
PY
```

## 10. Preserve the results

The complete result bundle is:

```text
$RESULTS_DIR/
  environment.txt
  torch_trt.log
  edge_onnx_export.log
  edge_onnx_export_s.txt
  edge_llm_build.log
  edge_llm_build_s.txt
  edge_visual_build.log
  edge_visual_build_s.txt
  edge_action_build.log
  edge_action_build_s.txt
  edge_inference.log
  edge_profile.json
  comparison.md
```

Copy this directory before rerunning section 1, since section 1 intentionally
removes the previous benchmark directory.
