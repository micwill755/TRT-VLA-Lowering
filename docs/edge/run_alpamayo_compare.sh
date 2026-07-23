#!/usr/bin/env bash
# Scripted version of compare_alpamayo_paths.md.
#
# Runs both Alpamayo paths on the same machine and writes a comparison report:
#   - Torch-TRT:  Test/vla/test_vla_alpamayo_e2e.py
#   - Edge:       ONNX export + engine builds + action_inference profiling
#
# Result bundle is written to $RESULTS_DIR (default:
# $HOME/tensorrt-edgellm-workspace/alpamayo_path_compare/results).
#
# Usage:
#   docs/edge/run_alpamayo_compare.sh
#
# Skip already-completed stages (useful when iterating) via env flags:
#   SKIP_TORCH=1 SKIP_ONNX=1 SKIP_BUILD=1 SKIP_INFER=1 docs/edge/run_alpamayo_compare.sh
#
# Every path below can be overridden from the environment.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Paths and prerequisites
# ---------------------------------------------------------------------------
export EDGE_LLM_ROOT=${EDGE_LLM_ROOT:-/home/micwilliams/workspace/TensorRT-Edge-LLM}
export TEST_ROOT=${TEST_ROOT:-/home/micwilliams/workspace/Test}
export ALPAMAYO_SRC=${ALPAMAYO_SRC:-/home/micwilliams/workspace/alpamayo/src}
export LEROBOT_PYTHON=${LEROBOT_PYTHON:-/home/micwilliams/miniforge3/envs/lerobot/bin/python}
export EDGELLM_EXPORT=${EDGELLM_EXPORT:-/home/micwilliams/miniforge3/envs/edgellm-export/bin/tensorrt-edgellm-export}

export WORKSPACE_DIR=${WORKSPACE_DIR:-$HOME/tensorrt-edgellm-workspace}
export MODEL_NAME=${MODEL_NAME:-Alpamayo-R1-10B}
export MODEL_DIR=${MODEL_DIR:-$WORKSPACE_DIR/$MODEL_NAME}
export SAMPLE_DIR=${SAMPLE_DIR:-$WORKSPACE_DIR/alpamayo_sample}
export COMPARE_DIR=${COMPARE_DIR:-$WORKSPACE_DIR/alpamayo_path_compare}
export RESULTS_DIR=${RESULTS_DIR:-$COMPARE_DIR/results}
export EDGE_LLM_PLUGIN_SO=${EDGE_LLM_PLUGIN_SO:-$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so}
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_PLUGIN_SO
export EDGE_BENCH_REQUESTS=${EDGE_BENCH_REQUESTS:-10}

SKIP_TORCH=${SKIP_TORCH:-0}
SKIP_ONNX=${SKIP_ONNX:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_INFER=${SKIP_INFER:-0}

log() { printf '\n=== %s ===\n' "$*"; }

log "Checking prerequisites"
test -x "$LEROBOT_PYTHON"
test -x "$EDGELLM_EXPORT"
test -f "$EDGE_LLM_PLUGIN_SO"
test -d "$MODEL_DIR"
for b in examples/llm/llm_build examples/multimodal/visual_build \
         examples/multimodal/action_build examples/multimodal/action_inference; do
    test -x "$EDGE_LLM_ROOT/build/$b"
done
test -f "$SAMPLE_DIR/input_action.json"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ---------------------------------------------------------------------------
# 1. Isolated benchmark directory + environment record
# ---------------------------------------------------------------------------
if [[ "$COMPARE_DIR" != "$WORKSPACE_DIR/alpamayo_path_compare" ]]; then
    echo "Refusing to remove unexpected COMPARE_DIR: $COMPARE_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR" "$COMPARE_DIR/onnx" "$COMPARE_DIR/engines/llm"

{
    date --iso-8601=seconds
    uname -a
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo "Test HEAD:     $(git -C "$TEST_ROOT" rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "Edge HEAD:     $(git -C "$EDGE_LLM_ROOT" rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "Alpamayo HEAD: $(git -C "$ALPAMAYO_SRC/.." rev-parse HEAD 2>/dev/null || echo n/a)"
} | tee "$RESULTS_DIR/environment.txt"

# ---------------------------------------------------------------------------
# 2. Torch-TRT e2e (compile + hot inference timers)
# ---------------------------------------------------------------------------
if [[ "$SKIP_TORCH" != "1" ]]; then
    log "Running Torch-TRT e2e"
    ( cd "$TEST_ROOT" && \
      PYTHONPATH="$ALPAMAYO_SRC:$TEST_ROOT" \
      EDGE_LLM_PLUGIN_SO="$EDGE_LLM_PLUGIN_SO" \
      "$LEROBOT_PYTHON" vla/test_vla_alpamayo_e2e.py ) \
      2>&1 | tee "$RESULTS_DIR/torch_trt.log"
else
    log "Skipping Torch-TRT e2e (SKIP_TORCH=1)"
fi

# ---------------------------------------------------------------------------
# 3. Fresh Edge ONNX export
# ---------------------------------------------------------------------------
if [[ "$SKIP_ONNX" != "1" ]]; then
    log "Exporting Edge ONNX"
    /usr/bin/time -f '%e' -o "$RESULTS_DIR/edge_onnx_export_s.txt" \
        env PYTHONPATH="$EDGE_LLM_ROOT" \
        "$EDGELLM_EXPORT" "$MODEL_DIR" "$COMPARE_DIR/onnx" \
            --max-kv-cache-capacity 4096 \
        2>&1 | tee "$RESULTS_DIR/edge_onnx_export.log"
    printf 'Edge ONNX export: %s s\n' \
        "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_onnx_export_s.txt")"
else
    log "Skipping Edge ONNX export (SKIP_ONNX=1)"
fi
test -f "$COMPARE_DIR/onnx/llm/model.onnx"
test -f "$COMPARE_DIR/onnx/visual/model.onnx"
test -f "$COMPARE_DIR/onnx/action/model.onnx"

# ---------------------------------------------------------------------------
# 4. Fresh Edge TensorRT engine builds
#    Stage tokenizer files first: llm_build's copyTokenizerFiles() is fatal and
#    the ONNX export does not emit tokenizer.json / tokenizer_config.json.
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" != "1" ]]; then
    log "Building Edge engines"
    cp "$MODEL_DIR/onnx/llm/tokenizer.json" "$COMPARE_DIR/onnx/llm/"
    cp "$MODEL_DIR/onnx/llm/tokenizer_config.json" "$COMPARE_DIR/onnx/llm/"

    cd "$EDGE_LLM_ROOT"
    /usr/bin/time -f '%e' -o "$RESULTS_DIR/edge_llm_build_s.txt" \
        ./build/examples/llm/llm_build \
            --onnxDir "$COMPARE_DIR/onnx/llm" \
            --engineDir "$COMPARE_DIR/engines/llm" \
            --maxInputLen 3424 --maxKVCacheCapacity 4096 --maxBatchSize 1 \
        2>&1 | tee "$RESULTS_DIR/edge_llm_build.log"

    /usr/bin/time -f '%e' -o "$RESULTS_DIR/edge_visual_build_s.txt" \
        ./build/examples/multimodal/visual_build \
            --onnxDir "$COMPARE_DIR/onnx/visual" \
            --engineDir "$COMPARE_DIR/engines" \
            --minImageTokens 160 --maxImageTokens 18432 --maxImageTokensPerImage 192 \
        2>&1 | tee "$RESULTS_DIR/edge_visual_build.log"

    /usr/bin/time -f '%e' -o "$RESULTS_DIR/edge_action_build_s.txt" \
        ./build/examples/multimodal/action_build \
            --onnxDir "$COMPARE_DIR/onnx/action" \
            --engineDir "$COMPARE_DIR/engines" \
            --maxBatchSize 1 \
        2>&1 | tee "$RESULTS_DIR/edge_action_build.log"
else
    log "Skipping Edge engine builds (SKIP_BUILD=1)"
fi
test -f "$COMPARE_DIR/engines/llm/llm.engine"
test -f "$COMPARE_DIR/engines/visual/visual.engine"
test -f "$COMPARE_DIR/engines/action/action.engine"
printf 'Edge LLM build:    %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_llm_build_s.txt")"
printf 'Edge visual build: %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_visual_build_s.txt")"
printf 'Edge action build: %s s\n' "$(tr -d '[:space:]' < "$RESULTS_DIR/edge_action_build_s.txt")"

# ---------------------------------------------------------------------------
# 5. Install runtime sidecars into the benchmark engine tree
# ---------------------------------------------------------------------------
log "Installing runtime sidecars"
cp "$MODEL_DIR/engines/llm/embedding.safetensors"       "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/llm/tokenizer.json"              "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/llm/tokenizer_config.json"       "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/llm/processed_chat_template.json" "$COMPARE_DIR/engines/llm/"
cp "$MODEL_DIR/engines/visual/preprocessor_config.json" "$COMPARE_DIR/engines/visual/"

# ---------------------------------------------------------------------------
# 6/7. Repeated Edge benchmark input
# ---------------------------------------------------------------------------
log "Creating repeated Edge benchmark input"
"$LEROBOT_PYTHON" - <<'PY'
import copy, json, os
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

# ---------------------------------------------------------------------------
# 8. Hot Edge inference profiling
# ---------------------------------------------------------------------------
if [[ "$SKIP_INFER" != "1" ]]; then
    log "Profiling Edge inference"
    cd "$EDGE_LLM_ROOT"
    ./build/examples/multimodal/action_inference \
        --engineDir "$COMPARE_DIR/engines/llm" \
        --multimodalEngineDir "$COMPARE_DIR/engines" \
        --inputFile "$COMPARE_DIR/edge_benchmark_input.json" \
        --outputFile "$COMPARE_DIR/edge_benchmark_output.json" \
        --dumpProfile \
        --profileOutputFile "$RESULTS_DIR/edge_profile.json" \
        --warmup=3 --noiseSeed=42 \
        2>&1 | tee "$RESULTS_DIR/edge_inference.log"
else
    log "Skipping Edge inference (SKIP_INFER=1)"
fi

# ---------------------------------------------------------------------------
# 9. Comparison report
# ---------------------------------------------------------------------------
log "Generating comparison report"
"$LEROBOT_PYTHON" - <<'PY'
import json, os, re
from pathlib import Path
results = Path(os.environ["RESULTS_DIR"])
requests = int(os.environ["EDGE_BENCH_REQUESTS"])
torch_log = (results / "torch_trt.log").read_text()
edge_profile = json.loads((results / "edge_profile.json").read_text())

def log_value(label):
    m = re.search(rf"^{re.escape(label)}:\s*([0-9.]+)", torch_log, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Missing {label!r} in torch_trt.log")
    return float(m.group(1))

def seconds_file(name):
    return float((results / name).read_text().strip())

def edge_stage(sid):
    for s in edge_profile.get("stages", []):
        if s.get("stage_id") == sid:
            return s
    raise RuntimeError(f"Missing Edge profile stage {sid!r}")

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
    torch_infer["vision"] + torch_infer["language_prefill"] + torch_infer["action_10_step_estimate"]
)

edge_ids = ("vision_encoder", "llm_prefill", "llm_generation", "action_inference")
edge_infer = {sid: float(edge_stage(sid)["total_gpu_time_ms"]) / requests for sid in edge_ids}
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
| Full action stage | ~{torch_infer['action_10_step_estimate']:.3f} (10x estimate) | {edge_infer['action_inference']:.3f} |

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

log "Done. Result bundle: $RESULTS_DIR"
