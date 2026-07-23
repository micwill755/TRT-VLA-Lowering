# Alpamayo Edge-LLM ONNX E2E Setup (x86)

End-to-end runbook for exporting Alpamayo-R1-10B to ONNX with TensorRT-Edge-LLM,
building TensorRT engines, preparing a sample `input_action.json`, and running
`action_inference` on x86.

This matches the workflow exercised on a local RTX 5090 (32 GiB) host.

## Prerequisites

- Access to gated Hugging Face assets:
  - [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
  - [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
  - [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (tokenizer / processor)
- Local checkouts:
  - `TensorRT-Edge-LLM` (built C++ examples + plugin)
  - `Test` (sample-input dump helper)
  - `alpamayo` (Physical AI AV loader)
- CUDA GPU with enough memory for engines + runtime buffers
  - Smoke test: rebuild with `--maxBatchSize 1` on ~32 GiB GPUs
  - Doc defaults use `--maxBatchSize 6` (needs more headroom)

Suggested paths used below:

```bash
export EDGE_LLM_ROOT=/home/micwilliams/workspace/TensorRT-Edge-LLM
export TEST_ROOT=/home/micwilliams/workspace/Test
export ALPAMAYO_SRC=/home/micwilliams/workspace/alpamayo/src
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
```

---

## 1. Create the Edge export conda env

Use a clean env for ONNX export (do not mix with Torch-TensorRT / Alpamayo eval envs).

```bash
conda create -n edgellm-export python=3.12 -y
conda activate edgellm-export

cd $EDGE_LLM_ROOT
pip install .

# Optional tools extra (quantize / LoRA / audio). Not required for Alpamayo FP16.
# pip install ".[tools]"

pip install -U "huggingface_hub[cli]"
hf auth login

export PYTHONPATH=$EDGE_LLM_ROOT:$PYTHONPATH
tensorrt-edgellm-export --help
```

Python 3.10 or 3.12 is supported. Prefer 3.12.

---

## 2. Download checkpoint and export ONNX

```bash
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

hf download nvidia/Alpamayo-R1-10B --local-dir "$MODEL_NAME"

conda activate edgellm-export
export PYTHONPATH=$EDGE_LLM_ROOT:$PYTHONPATH

tensorrt-edgellm-export \
  "$MODEL_NAME" \
  "$MODEL_NAME/onnx" \
  --max-kv-cache-capacity 4096
```

Expected layout:

```text
$MODEL_NAME/onnx/
  llm/model.onnx (+ model.onnx.data, embedding.safetensors, config.json, ...)
  visual/model.onnx (+ model.onnx.data, config.json, ...)
  action/model.onnx (+ model.onnx.data, config.json, ...)
```

Notes:

- Only FP16 is supported for Alpamayo in current Edge-LLM release.
- `--max-kv-cache-capacity` must match `--maxKVCacheCapacity` used in `llm_build`.
- Small `.onnx` sizes are normal; large weights are externalized.

### Known sidecar gaps (fix before engine build / inference)

On this host, export sometimes omitted runtime sidecars. Ensure these exist before
running inference (create them if missing):

| File | Where it must end up |
|------|----------------------|
| `embedding.safetensors` | `engines/llm/` (source: `onnx/llm/`) |
| `tokenizer.json`, `tokenizer_config.json` | `engines/llm/` |
| `preprocessor_config.json` | `engines/visual/` (and ideally `onnx/visual/`) |
| `processed_chat_template.json` | `engines/llm/` **and** `onnx/llm/` (see below; do not trust the fallback) |

**Chat template (important):** Alpamayo HF has no tokenizer/`chat_template`, so
export often writes the *fallback* `processed_chat_template.json` with
`"content_types": {}` and plain `User:` / `Assistant:` roles. That causes
`Unknown content type: image` at runtime and empty `output_text`.

Overwrite both copies with the Qwen3-VL + Alpamayo CoT template:

```bash
python - <<'PY'
import json
from pathlib import Path

tmpl = {
    "model_path": "Alpamayo-R1-10B",
    "roles": {
        "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
        "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
        "assistant": {"prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>\n"},
    },
    "content_types": {
        "image": {"format": "<|vision_start|><|image_pad|><|vision_end|>"},
        "video": {"format": "<|vision_start|><|video_pad|><|vision_end|>"},
    },
    "generation_prompt": "<|im_start|>assistant\n<|cot_start|>",
    "default_system_prompt": "",
}
root = Path.home() / "tensorrt-edgellm-workspace" / "Alpamayo-R1-10B"
for dest in (root / "onnx" / "llm", root / "engines" / "llm"):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "processed_chat_template.json"
    path.write_text(json.dumps(tmpl, indent=2) + "\n")
    print("wrote", path)
PY
```

Quick restore helpers (use `lerobot` env for tokenizer/processor; it has Pillow /
torchvision / `physical_ai_av` deps):

```bash
# Embedding
cp -n "$WORKSPACE_DIR/$MODEL_NAME/onnx/llm/embedding.safetensors" \
      "$WORKSPACE_DIR/$MODEL_NAME/engines/llm/"

# Tokenizer (Alpamayo = Qwen3-VL tokenizer + trajectory tokens)
/home/micwilliams/miniforge3/envs/lerobot/bin/python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer

root = Path.home() / "tensorrt-edgellm-workspace" / "Alpamayo-R1-10B"
cfg = json.loads((root / "config.json").read_text())
vlm = cfg.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
tok = AutoTokenizer.from_pretrained(vlm, trust_remote_code=True)
tok.add_tokens([f"<i{v}>" for v in range(cfg.get("traj_vocab_size", 768))])
tok.add_tokens([
    "<|traj_history|>", "<|traj_future|>",
    "<|traj_history_start|>", "<|traj_future_start|>",
    "<|traj_history_end|>", "<|traj_future_end|>",
], special_tokens=True)
for dest in (root / "engines" / "llm", root / "onnx" / "llm"):
    dest.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(dest)
    print("saved tokenizer to", dest)
PY

# Visual preprocessor
/home/micwilliams/miniforge3/envs/lerobot/bin/python - <<'PY'
import json, shutil
from pathlib import Path
from transformers import AutoProcessor

root = Path.home() / "tensorrt-edgellm-workspace" / "Alpamayo-R1-10B"
cfg = json.loads((root / "config.json").read_text())
vlm = cfg.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
proc = AutoProcessor.from_pretrained(
    vlm, trust_remote_code=True,
    min_pixels=128 * 28 * 28,
    max_pixels=2048 * 32 * 32,
    size={"longest_edge": 16777216, "shortest_edge": 65536},
)
for dest in (root / "onnx" / "visual", root / "engines" / "visual"):
    dest.mkdir(parents=True, exist_ok=True)
    proc.save_pretrained(dest)
    src = dest / "processor_config.json"
    dst = dest / "preprocessor_config.json"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    print("saved preprocessor to", dest)
PY
```

---

## 3. Build Edge-LLM C++ binaries (once)

From the Edge-LLM build tree:

```bash
cd $EDGE_LLM_ROOT/build
cmake --build . --target llm_build visual_build action_build action_inference NvInfer_edgellm_plugin -j$(nproc)
```

Confirm:

```bash
ls $EDGE_LLM_ROOT/build/examples/llm/llm_build
ls $EDGE_LLM_ROOT/build/examples/multimodal/{visual_build,action_build,action_inference}
ls $EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
```

Always set:

```bash
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
```

---

## 4. Build TensorRT engines

Use **`--maxBatchSize 1`** for a single-request smoke test on 32 GiB GPUs.
Doc defaults use `6` and can OOM during runtime buffer allocation even if weights fit.

```bash
cd $EDGE_LLM_ROOT

./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxInputLen 3424 \
  --maxKVCacheCapacity 4096 \
  --maxBatchSize 1

./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 160 \
  --maxImageTokens 18432 \
  --maxImageTokensPerImage 192

./build/examples/multimodal/action_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/action \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1
```

Expected:

```text
$MODEL_NAME/engines/
  llm/llm.engine
  visual/visual.engine
  action/action.engine
```

After build, re-apply any missing sidecars from section 2 (`embedding.safetensors`,
tokenizer files, `preprocessor_config.json`, **and** the fixed
`processed_chat_template.json`).

---

## 5. Dump a real sample `input_action.json`

The docs use placeholder paths (`/path/to/frame_XX.png`). Those files do not
exist. Use the Test helper to pull the same Physical AI AV clip as
`test_vla_alpamayo_e2e.py` and write Edge-compatible JSON.

Requires an env with `physical_ai_av` (on this machine: `lerobot`), **not**
`edgellm-export`.

Without activating conda:

```bash
PYTHONPATH=$ALPAMAYO_SRC:$TEST_ROOT \
  /home/micwilliams/miniforge3/envs/lerobot/bin/python \
  $TEST_ROOT/trt/dump_alpamayo_edge_input.py \
  --output-dir $WORKSPACE_DIR/alpamayo_sample
```

This writes:

```text
$WORKSPACE_DIR/alpamayo_sample/
  frames/frame_00.png ... frame_15.png
  input_action.json
  gt.json                  # real minADE GT (see alpamayo_vla_pipeline_pytest.md)
```

Default clip matches the e2e script:
`030c760c-ae38-49aa-9ad8-f5650a545d26` at `t0_us=5100000`.

For the VLA pytest harness, dump with `--num-traj-samples 6` and copy into
`$EDGELLM_DATA_DIR` as described in
[alpamayo_vla_pipeline_pytest.md](./alpamayo_vla_pipeline_pytest.md).

---

## 6. Run `action_inference`

```bash
cd $EDGE_LLM_ROOT
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so

./build/examples/multimodal/action_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/alpamayo_sample/input_action.json \
  --outputFile $WORKSPACE_DIR/output_action.json \
  --dumpOutput
```

Success looks like:

- LLM / visual / action engines load
- Tokenizer loads chat template from `engines/llm/processed_chat_template.json`
- No `Unknown content type: image` warnings
- Inference completes
- `$WORKSPACE_DIR/output_action.json` has non-empty `output_text` and `output_trajectory`
- With `--dumpOutput`, `formatted_complete_request` includes
  `<|vision_start|><|image_pad|><|vision_end|>` (16× for the sample) and
  ends with `<|im_start|>assistant\n<|cot_start|>`

Optional profiling:

```bash
./build/examples/multimodal/action_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/alpamayo_sample/input_action.json \
  --outputFile $WORKSPACE_DIR/output_action.json \
  --dumpProfile --warmup=3
```

---

## 7. CI / accuracy path (optional)

Edge-LLM also has pytest coverage in:

- `TensorRT-Edge-LLM/tests/defs/test_vla_pipeline.py`

That path expects internal datasets under `$EDGELLM_DATA_DIR`:

```text
$EDGELLM_DATA_DIR/updated_datasets/alpamayo_eval_dataset/input.json
$EDGELLM_DATA_DIR/updated_datasets/alpamayo_action_chat/input.json
```

and reports minADE. Those datasets are not required for the smoke test above.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Failed to load image: /path/to/frame_00.png` | Still using doc placeholders. Run section 5 dump helper. |
| `Embedding file not found: .../engines/llm/embedding.safetensors` | Copy from `onnx/llm/`. |
| Tokenizer load failure / missing `tokenizer.json` | Regenerate tokenizer (section 2). |
| `Failed to open preprocessor_config.json` | Save Qwen3-VL processor into `engines/visual/` (section 2). |
| `cudaMalloc ... out of memory` after engines load | Engines built with `maxBatchSize=6`. Rebuild LLM/action with `--maxBatchSize 1`. |
| `Failed to import physical_ai_av` | Dump script must use `lerobot` (or another Alpamayo) env, not `edgellm-export`. |
| Plugin not found | Export `EDGELLM_PLUGIN_PATH` to `libNvInfer_edgellm_plugin.so`. |
| `Unknown content type: image` (×N) | Fallback chat template with empty `content_types`. Write the Qwen3-VL template in section 2. |
| Empty `output_text` but trajectory present | Same as above — vision tokens never entered the prompt. |

---

## Architecture reminder

```text
HF Alpamayo-R1-10B
        |
        v
tensorrt-edgellm-export
        |
   +----+----+
   v    v    v
 llm  visual action   (.onnx + sidecars)
   |    |    |
   v    v    v
 llm_build / visual_build / action_build
   |    |    |
   v    v    v
 llm.engine / visual.engine / action.engine
               |
               v
        action_inference
               |
               v
     output_text + output_trajectory
```

Do **not** use `vla_inference` for Alpamayo. That binary targets the
PI0.5/GR00T `VlaInferenceRuntime` path. Alpamayo uses `action_inference` +
`LLMInferenceRuntime` + `Alpamayo1ActionRunner`.

---

## Related files

- Edge docs: `TensorRT-Edge-LLM/docs/source/user_guide/examples/vla.md`
- Dump helper: `Test/trt/dump_alpamayo_edge_input.py`
- Torch-TRT parity script: `Test/vla/test_vla_alpamayo_e2e.py`
- Edge CI tests: `TensorRT-Edge-LLM/tests/defs/test_vla_pipeline.py`
- Local pytest harness runbook: [alpamayo_vla_pipeline_pytest.md](./alpamayo_vla_pipeline_pytest.md)
