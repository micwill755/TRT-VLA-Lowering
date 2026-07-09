# Torch-TRT Pipelines

Export, inference, and benchmark orchestration for VLA models on top of [TensorRT Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/).

The CLI entry point is `app.py`. It builds an `EdgeContext` from a model profile and sample dataset frame, then runs the export and/or benchmark pipelines.

## Prerequisites

- Python 3.10+ with PyTorch, Torch-TensorRT, and LeRobot dependencies (e.g. `lerobot` conda env)
- CUDA GPU
- Built Edge-LLM TensorRT plugin (required for export **and** C++ inference):

```bash
export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so   # Python export
export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so    # C++ llm_inference
```

The plugin `.so` must match the TensorRT major version of the `llm_inference` binary (e.g. TRT 11 plugin with `build-plugin-trt11/.../llm_inference`).

## Quick start

From this directory:

```bash
cd /home/micwilliams/workspace/Test
```

### GR00T (four engines: vision, language, action_context, action)

```bash
# Export + benchmark (default)
python app.py --model gr00t --device cuda --engine-dir /tmp/groot_edge_llm

# Export and benchmark separately
python app.py --model gr00t --export-only --device cuda --engine-dir /tmp/groot_edge_llm
python app.py --model gr00t --benchmark-only --device cuda --engine-dir /tmp/groot_edge_llm

# Eager inference only
python app.py --model gr00t --inference-only --device cuda
```

### Pi0.5 (three engines: vision, language, action)

```bash
python app.py --model pi05 --device cuda --engine-dir /tmp/pi05_edge_llm
python app.py --model pi05 --export-only --device cuda --engine-dir /tmp/pi05_edge_llm
python app.py --model pi05 --benchmark-only --device cuda --engine-dir /tmp/pi05_edge_llm
python app.py --model pi05 --inference-only --device cuda
```

### SmolVLA (three engines: vision, language, action)

```bash
python app.py --model smolvla --device cuda --engine-dir /tmp/smolvla_edge_llm
python app.py --model smolvla --export-only --device cuda --engine-dir /tmp/smolvla_edge_llm
python app.py --model smolvla --benchmark-only --device cuda --engine-dir /tmp/smolvla_edge_llm
python app.py --model smolvla --inference-only --device cuda
```

### MolmoAct2 (e2e parity script; `app.py` pipeline in progress)

```bash
python test_vla_molmo2_e2e.py
```

### Alpamayo (e2e parity script; requires Alpamayo Python 3.12 env)

```bash
python test_vla_alpamayo_e2e.py
```

## Useful flags

| Flag | Description |
|------|-------------|
| `--model` | VLA profile: `gr00t`, `pi05`, `smolvla`, `molmo2` |
| `--model-id` | Hugging Face checkpoint override |
| `--engine-dir` | Where engines are written/read (default varies by model) |
| `--device` | `cuda` or `cpu` |
| `--dataset-id` | LeRobot dataset (default: `lerobot/libero`) |
| `--episode-index` | Dataset episode (default: `0`) |
| `--frame-index` | Frame within episode (default: `0`) |
| `--export-only` | Export engines; skip benchmark |
| `--benchmark-only` | Benchmark only; skip export |
| `--inference-only` | Single eager inference pass |
| `--parity-mode` | Stage parity: `e2e`, `isolated`, or `both` (default) |

## Engine layout

**GR00T** (`/tmp/groot_edge_llm/`):

```text
visual/visual.engine
language/language.engine
action_context/context.engine
action/action.engine
```

**Pi0.5 / SmolVLA** (`/tmp/pi05_edge_llm/`, `/tmp/smolvla_edge_llm/`):

```text
visual/visual.engine
language/language.engine
action/action.engine
```

## Edge LLM runtime (C++)

Exported engines are loaded by TensorRT Edge-LLM's `llm_inference` binary. Point `--engineDir` at the **language** subdirectory and `--multimodalEngineDir` at the export root.

**Binary path** (TRT 11 build):

```text
/home/micwilliams/workspace/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference
```

Set `EDGELLM_PLUGIN_PATH` before running — without it, engine load fails with `Cannot find plugin: AttentionPlugin`.

### GR00T

```bash
# Export
python app.py --model gr00t --export-only --device cuda --engine-dir /tmp/groot_edge_llm

# C++ inference
export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so
/home/micwilliams/workspace/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference \
  --engineDir=/tmp/groot_edge_llm/language \
  --multimodalEngineDir=/tmp/groot_edge_llm \
  --inputFile=/tmp/groot_edge_llm/runtime_smoke/input_action.json \
  --outputFile=/tmp/groot_edge_llm/runtime_smoke/output_e2e.json \
  --maxGenerateLength=0 \
  --dumpOutput \
  --dumpProfile
```

### Pi0.5

```bash
# Export
python app.py --model pi05 --export-only --device cuda --engine-dir /tmp/pi05_edge_llm

# C++ inference
export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so
/home/micwilliams/workspace/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference \
  --engineDir=/tmp/pi05_edge_llm/language \
  --multimodalEngineDir=/tmp/pi05_edge_llm \
  --inputFile=/tmp/pi05_edge_llm/runtime_smoke/input_action.json \
  --outputFile=/tmp/pi05_edge_llm/runtime_smoke/output_e2e.json \
  --maxGenerateLength=0 \
  --dumpOutput \
  --dumpProfile
```

### SmolVLA

```bash
# Export
python app.py --model smolvla --export-only --device cuda --engine-dir /tmp/smolvla_edge_llm

# C++ inference
export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so
/home/micwilliams/workspace/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference \
  --engineDir=/tmp/smolvla_edge_llm/language \
  --multimodalEngineDir=/tmp/smolvla_edge_llm \
  --inputFile=/tmp/smolvla_edge_llm/runtime_smoke/input_action.json \
  --outputFile=/tmp/smolvla_edge_llm/runtime_smoke/output_e2e.json \
  --maxGenerateLength=0 \
  --dumpOutput \
  --dumpProfile
```

See `docs/source/edge_llm/e2e.rst` for request JSON format, tokenization parity notes, and common failure modes.

## Performance metrics

Per-engine latency (ms) from `test_vla_*_e2e.py` on a single libero frame. **eager** = PyTorch forward pass; **edge** = Torch-TensorRT compiled engine (same script, warmed up over 100 iterations per stage).

| Model | Stage | eager (ms) | edge (ms) |
|-------|-------|------------|-----------|
| **Pi0.5** | Vision | 5.83 | 3.57 |
| | Language | 15.11 | 13.91 |
| | Action (full) | 128.2 | 21.1 |
| | **E2E** | 33.8¹ | 19.6¹ |
| **GR00T** | Vision | 7.51 | 3.54 |
| | Language | 8.16 | 4.93 |
| | Action context | 1.49 | 1.50 |
| | Action (full) | 17.7 | 7.7 |
| | **E2E** | 21.2¹ | 11.5¹ |
| **SmolVLA** | Vision | 7.77 | 3.26 |
| | Language | 5.90 | 0.95 |
| | Action (full) | 97.5 | 8.4 |
| | **E2E** | 23.4¹ | 5.0¹ |

¹ **E2E** in the parity scripts = vision + language + **one** diffusion step (not the full denoising loop). **Action (full)** = per-step time × `num_inference_steps` (10 for Pi0.5/SmolVLA, 4 for GR00T) + action context for GR00T.

C++ `llm_inference` end-to-end latency with `--warmup=10` (full deployment path, libero smoke JSON):

| Model | Vision | Language prefill | Action (Diffusor) | E2E |
|-------|--------|------------------|-------------------|-----|
| Pi0.5 | 2.78 ms | 23.99 ms | 26.38 ms | 53.4 ms |
| GR00T | 3.44 ms | 7.49 ms | 110.34 ms | 121.4 ms |
| SmolVLA | 6.14 ms | 3.83 ms | 9.15 ms | 19.4 ms |

Reproduce C++ timings:

```bash
export EDGELLM_PLUGIN_PATH=/home/micwilliams/workspace/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/libNvInfer_edgellm_plugin.so

/home/micwilliams/workspace/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference \
  --engineDir=/tmp/<model>_edge_llm/language \
  --multimodalEngineDir=/tmp/<model>_edge_llm \
  --inputFile=/tmp/<model>_edge_llm/runtime_smoke/input_action.json \
  --outputFile=/tmp/<model>_edge_llm/runtime_smoke/output_e2e.json \
  --maxGenerateLength=0 --warmup=10 --dumpProfile
```

## Benchmark output

Benchmark runs inference once per mode (`eager`, `in_memory`, `serialized`) and prints per-stage parity tables, for example:

```text
Parity: eager vs in_memory
--------------------------------------------------------------------------------
Stage            Tensor           Mean Abs    Max Abs   Rel L2  Rel Mean   Close
...
```

## Documentation

Sphinx docs live under `docs/source/`. Build HTML:

```bash
cd docs && make html
```

Open `docs/build/html/index.html` for architecture, per-model examples, and customization guides.

## Project layout

```text
app.py                          CLI entry
test_vla_*_e2e.py               Per-model TRT parity scripts
trt/orchestrator/               EdgeOrchestrator
trt/pipelines/                  Export, inference, benchmark pipelines
trt/executor/models/            Per-model stage hooks (groot, pi05, smolvla)
trt/modules/export/             Reusable export modules (vision, language, diffusion)
docs/source/examples/           Per-model example pages
```
