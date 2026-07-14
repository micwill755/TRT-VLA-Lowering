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

```bash
sudo docker stop $(sudo docker ps -q) 2>/dev/null
sudo bash -c 'echo 10240 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'
sudo swapon /gtl/swapfile 2>/dev/null || true
./docker/run.sh python3 vla/test_vla_pi05_e2e.py
```

## DRIVE AGX Thor (DriveOS)

Pi0.5 TRT compile on **DRIVE AGX Thor** needs extra host setup (GPU carveout,
swap, engine output path). Desktop GPUs (e.g. RTX 5090) do not need these steps.

| Topic | Guide |
|-------|-------|
| Docker + Thor stack | [docker/RUNBOOK.md](docker/RUNBOOK.md) |
| GPU carveout (hugetlbfs) | [RUNBOOK — GPU memory carveout](docker/RUNBOOK.md#gpu-memory-carveout-thor--required-for-pi05) |
| Host swap file | [RUNBOOK — Host swap file](docker/RUNBOOK.md#host-swap-file-thor--required-for-language-trt-compile) |
| Pi0.5 compile RAM/GPU table | [RUNBOOK — Pi0.5 compile memory requirements](docker/RUNBOOK.md#pi05-compile-memory-requirements-thor-test_vla_pi05_e2epy) |

Quick Thor compile checklist:

```bash
# Host: 20 GB GPU carveout + swap + engine dir on /gtl
sudo bash -c 'echo 10240 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'
# swap: see docker/RUNBOOK.md if not already configured

cd /path/to/TRT-VLA-Lowering
cp docker/env.example docker/.env   # ENGINE_DIR=/gtl/pi05_edge_llm
./docker/run.sh python3 vla/test_vla_pi05_e2e.py
```

### Desktop parity container (RTX 5090 / x86_64)

Native conda on 5090 (torch **2.14**, TRT **11**) is **not** the same stack as Thor
(torch **2.10**, TRT **10.14**). For a fair comparison, use the desktop Docker
image that pins the **same versions as Thor**:

| | Thor (`run.sh`) | Desktop parity (`run-desktop.sh`) | Native 5090 conda |
|---|---|---|---|
| PyTorch | 2.10+cu130 | 2.10+cu130 | 2.14+cu132 |
| TensorRT | 10.14 (host mount) | 10.14 (pip) | 11.0 |
| torch-tensorrt | 2.10 | 2.10 | 2.14 |
| `TRT_VLA_THOR` | `1` (cuDNN off) | `0` default; set `1` to match Thor | unset |
| Edge-LLM plugin | TRT **10.14** aarch64 build | TRT **10.14** x86_64 build | TRT **11** build |

On the 5090 machine:

```bash
cd ~/workspace/Test/TRT-VLA-Lowering   # or your clone
git checkout thor                      # same branch as Thor

cp docker/env.desktop.example docker/.env
# Edit: TRT_VLA_ROOT, LEROBOT_ROOT, EDGE_LLM_PLUGIN_SO (x86 TRT 10.14 plugin)

chmod +x docker/build-desktop.sh docker/run-desktop.sh
./docker/build-desktop.sh
./docker/run-desktop.sh bash

# Inside container — verify Thor-matched stack
python3 -c "import torch, torch_tensorrt, tensorrt as trt; print(torch.__version__, torch_tensorrt.__version__, trt.__version__)"
# expect: 2.10.0+cu130  2.10.0  10.14.x

python3 vla/test_vla_pi05_e2e.py
```

Compare cuDNN behavior inside the **same** container:

```bash
TRT_VLA_THOR=0 ./docker/run-desktop.sh python3 vla/test_vla_pi05_e2e.py
TRT_VLA_THOR=1 ./docker/run-desktop.sh python3 vla/test_vla_pi05_e2e.py
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`docker run --gpus all`). See `docker/RUNBOOK.md` for the Edge-LLM plugin x86 build notes.

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
