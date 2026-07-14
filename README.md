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

The plugin `.so` must match the TensorRT major version you compile against:

| Target | TensorRT | Plugin build dir |
|--------|----------|------------------|
| Thor Docker / DriveOS | **10.14** | `build-plugin-trt10/` (aarch64) |
| Desktop parity Docker (5090) | **10.14** | `build-plugin-trt10/` (x86_64) |
| Native 5090 conda dev | **11** | `build-plugin-trt11/` (x86_64) |

See [Building the Edge-LLM plugin](#building-the-edge-llm-plugin-libnvinfer_edgellm_pluginso) below.

## Building the Edge-LLM plugin (`libNvInfer_edgellm_plugin.so`)

VLA export scripts call `load_plugins_for_trt()` and require a native TensorRT
plugin built from [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).
The pip `tensorrt` wheel does **not** ship dev headers (`NvInfer.h`) — you need
the full TensorRT tarball for the plugin build.

Clone Edge-LLM (same repo/branch used on Thor, e.g. `cosmos`):

```bash
git clone git@github.com:NVIDIA/TensorRT-Edge-LLM.git   # or your fork
cd TensorRT-Edge-LLM
git submodule update --init 3rdParty/nlohmannJson
```

`ENABLE_CUTE_DSL=OFF` is enough for Pi0.5 / GR00T / SmolVLA VLA export (attention
plugins only). Use `ALL` only if you need Qwen3.5 GDN / NVFP4 MoE runtime paths.

### Thor (aarch64, DriveOS 7.0.5, TRT 10.14)

TensorRT dev headers are not installed system-wide on Thor — point `TRT_PACKAGE_DIR`
at the full TRT package (headers + libs). On DriveOS this is typically under `/gtl`:

```bash
cd /path/to/TensorRT-Edge-LLM
rm -rf build-plugin-trt10 && mkdir build-plugin-trt10 && cd build-plugin-trt10
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/gtl/managed/builds/TensorRT-10.14.2.2 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=auto-thor \
  -DCUDA_CTK_VERSION=13.0 \
  -DENABLE_CUTE_DSL=OFF
make NvInfer_edgellm_plugin -j"$(nproc)"
```

Output:

```text
build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0
```

Set in `docker/thor/.env`:

```bash
EDGE_LLM_PLUGIN_SO=/home/mwilliams/tensorrt-edge-llm/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0
```

### Desktop parity (x86_64, RTX 5090, TRT 10.14)

The plugin **must** link `libnvinfer.so.10` (TRT 10.x). If your host only has
TensorRT 11 (`libnvinfer.so.11`), a host build silently links the wrong version
and fails inside the container with `libnvinfer.so.11: cannot open shared object`.

**Recommended: build inside the desktop container.** It already has the pip
TensorRT 10.14 libs, so the plugin links the exact runtime it will use. Headers
come from the open-source TensorRT repo (no gated NVIDIA download):

```bash
cd ~/workspace/Test

# Rebuild the desktop image first (adds cmake + build-essential):
./docker/desktop/build.sh

# Build the plugin in-container (clones TRT 10.14 headers, uses pip libs):
EDGE_LLM_SRC=~/workspace/TensorRT-Edge-LLM ./docker/desktop/build-plugin.sh
```

This writes `build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0` in the Edge-LLM
source and prints the `ldd` line (expect `libnvinfer.so.10`).

**Alternative: host build against a TRT 10.14 tarball.** If you have the
[TensorRT 10.14 GA](https://developer.nvidia.com/tensorrt) x86_64 tarball (CUDA 13)
extracted, keep TRT 11 off the linker path and point cmake at it:

```bash
cd ~/workspace/TensorRT-Edge-LLM
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
unset LD_LIBRARY_PATH
rm -rf build-plugin-trt10 && mkdir build-plugin-trt10 && cd build-plugin-trt10
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=$HOME/workspace/TensorRT-10.14.2.2 \
  -DCUDA_CTK_VERSION=13.0 \
  -DENABLE_CUTE_DSL=OFF
make NvInfer_edgellm_plugin -j"$(nproc)"
```

Verify (either method):

```bash
ldd build-plugin-trt10/libNvInfer_edgellm_plugin.so.* | grep nvinfer   # must be .so.10
strings build-plugin-trt10/libNvInfer_edgellm_plugin.so.* | grep -E 'AttentionPlugin|ViTAttentionPlugin'
```

Set in `docker/desktop/.env`:

```bash
EDGE_LLM_PLUGIN_SO=/home/micwilliams/workspace/TensorRT-Edge-LLM/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0
```

`docker/desktop/run.sh` auto-resolves `.so.1` vs `.so.1.0`, mounts the plugin dir,
and stages a world-readable copy into the container.

### Native dev (x86_64, TRT 11)

For the native 5090 conda stack (torch 2.14 / TRT 11), build into `build-plugin-trt11`
and point `TRT_PACKAGE_DIR` at your TRT 11 install:

```bash
cd ~/workspace/TensorRT-Edge-LLM
rm -rf build-plugin-trt11 && mkdir build-plugin-trt11 && cd build-plugin-trt11
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT-11.x \
  -DENABLE_CUTE_DSL=OFF
make NvInfer_edgellm_plugin -j"$(nproc)"
```

```bash
export EDGE_LLM_PLUGIN_SO=/path/to/build-plugin-trt11/libNvInfer_edgellm_plugin.so.1.0
export EDGELLM_PLUGIN_PATH="$EDGE_LLM_PLUGIN_SO"
```

More Thor/Docker troubleshooting: [docker/RUNBOOK.md](docker/RUNBOOK.md).

## Docker layout

```text
docker/
├── common/entrypoint.sh, requirements.container.txt
├── thor/          # DRIVE AGX Thor (aarch64, TRT 10.14 host mount)
├── desktop/       # RTX 5090 parity (x86_64, TRT 10.14 pip)
├── build.sh       # wrapper → thor/build.sh
├── run.sh         # wrapper → thor/run.sh
└── RUNBOOK.md
```

## Running Pi0.5 e2e (`test_vla_pi05_e2e.py`)

Primary parity script: `vla/test_vla_pi05_e2e.py`. It compiles vision / language /
diffusion TRT engines, checks `close%=100` parity per stage, and prints eager vs
TRT timings (100 warmed iterations per stage).

| Target | When to use |
|--------|-------------|
| **Thor Docker** | Deployment target (DriveOS 7.0.5, TRT 10.14) |
| **5090 parity Docker** | Pre-validate Thor behavior on a desktop GPU (same torch/TRT as Thor) |
| **5090 native conda** | Fastest local dev (torch 2.14, TRT 11) — **not** comparable to Thor |

### DRIVE AGX Thor

**Stack:** PyTorch 2.10+cu130, torch-tensorrt 2.10, TensorRT 10.14 (host mount),
`TRT_VLA_THOR=1` (cuDNN off), Edge-LLM plugin `build-plugin-trt10/` (aarch64).

**One-time setup:**

```bash
cd /home/mwilliams/test/TRT-VLA-Lowering   # or your clone

cp docker/thor/env.example docker/thor/.env
# Edit docker/thor/.env:
#   LEROBOT_ROOT=/home/mwilliams/test/lerobot
#   EDGE_LLM_PLUGIN_SO=/home/mwilliams/tensorrt-edge-llm/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0
#   ENGINE_DIR=/gtl/pi05_edge_llm

chmod +x docker/thor/build.sh docker/thor/run.sh
./docker/thor/build.sh
```

Build the Edge-LLM plugin on Thor if needed — see
[Thor (aarch64, DriveOS 7.0.5, TRT 10.14)](#thor-aarch64-driveos-705-trt-1014) above.

| Topic | Guide |
|-------|-------|
| Full Thor runbook | [docker/RUNBOOK.md](docker/RUNBOOK.md) |
| GPU carveout | [RUNBOOK — GPU memory carveout](docker/RUNBOOK.md#gpu-memory-carveout-thor--required-for-pi05) |
| Host swap | [RUNBOOK — Host swap file](docker/RUNBOOK.md#host-swap-file-thor--required-for-language-trt-compile) |
| Compile memory | [RUNBOOK — Pi0.5 compile memory](docker/RUNBOOK.md#pi05-compile-memory-requirements-thor-test_vla_pi05_e2epy) |

**Host prep (each boot / before compile):** Pi0.5 language TRT needs ~20 GB GPU
carveout, host swap, and engine output on `/gtl`. Details:
[docker/RUNBOOK.md](docker/RUNBOOK.md).

```bash
sudo docker stop $(sudo docker ps -q) 2>/dev/null
sudo bash -c 'echo 10240 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'
sudo swapon /gtl/swapfile 2>/dev/null || true
```

**Run:**

```bash
cd /home/mwilliams/test/TRT-VLA-Lowering
./docker/thor/run.sh python3 vla/test_vla_pi05_e2e.py
```

Interactive shell: `./docker/thor/run.sh bash` then `python3 vla/test_vla_pi05_e2e.py`.

Legacy wrappers `./docker/run.sh` and `./docker/build.sh` delegate to `thor/`.

### RTX 5090 — parity Docker (matches Thor)

**Stack:** Same versions as Thor (torch 2.10+cu130, TRT 10.14 pip, torch-tensorrt 2.10).
Use this to preview Thor compile behavior and timings before running on hardware.

**One-time setup:**

```bash
cd ~/workspace/Test   # repo root (directory containing docker/, trt/, vla/)

git checkout thor
cp docker/desktop/env.example docker/desktop/.env
# Edit docker/desktop/.env:
#   TRT_VLA_ROOT=/home/micwilliams/workspace/Test
#   LEROBOT_ROOT=/home/micwilliams/workspace/lerobot
#   EDGE_LLM_PLUGIN_SO=/home/micwilliams/workspace/TensorRT-Edge-LLM/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0

chmod +x docker/desktop/build.sh docker/desktop/run.sh docker/desktop/build-plugin.sh
./docker/desktop/build.sh
```

Build the TRT **10.14** plugin inside the container (avoids host TRT 11 `libnvinfer.so.11` mismatch):

```bash
EDGE_LLM_SRC=~/workspace/TensorRT-Edge-LLM ./docker/desktop/build-plugin.sh
# verify: ldd .../libNvInfer_edgellm_plugin.so.* | grep nvinfer  → libnvinfer.so.10
```

**Run:**

```bash
./docker/desktop/run.sh python3 vla/test_vla_pi05_e2e.py
```

`run.sh` sets pip `nvidia/cu13` + `nvidia/cudnn` on `LD_LIBRARY_PATH` (required on
Blackwell / sm_120 — base-image cuBLAS 13.0 otherwise breaks large GEMMs).

Optional — match Thor cuDNN-disabled behavior:

```bash
TRT_VLA_THOR=1 ./docker/desktop/run.sh python3 vla/test_vla_pi05_e2e.py
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### RTX 5090 — native conda (TRT 11 dev)

**Stack:** torch 2.14+cu132, TRT 11, torch-tensorrt 2.14, plugin `build-plugin-trt11/`.
Fastest on 5090; use for local iteration, not Thor deployment prediction.

```bash
conda activate lerobot
export EDGE_LLM_PLUGIN_SO=/path/to/build-plugin-trt11/libNvInfer_edgellm_plugin.so.1.0
cd /path/to/TRT-VLA-Lowering
python3 vla/test_vla_pi05_e2e.py
```

### Pi0.5 timing comparison (RTX 5090, single libero frame)

From `test_vla_pi05_e2e.py` output (`* execute` = per-stage forward; `total` = vision + language + one diffusion step). Eager numbers match across stacks; TRT differences are mostly **language**.

| Stage | Eager (both) | TRT parity Docker (TRT 10.14) | TRT native conda (TRT 11) |
|-------|----------------|----------------------------------|---------------------------|
| Vision | ~16.5 ms | 9.0 ms (**1.8×**) | 9.3 ms (**1.8×**) |
| Language | ~15.0 ms | 35.1 ms (**0.4×**) | 14.0 ms (**1.1×**) |
| Diffusion | ~12.5 ms | 2.2 ms (**5.6×**) | 2.1 ms (**6.0×**) |
| **Total** | ~44 ms | 46 ms (**0.95×**) | 25 ms (**1.7×**) |

**Takeaways:**

- **Eager ~identical** → model/data path is the same; container setup is correct.
- **Vision / diffusion TRT** are similar between TRT 10.14 container and TRT 11 native.
- **Language TRT** is ~2.5× slower in the TRT 10.14 parity container than TRT 11 native; this drives the total gap.
- Use **parity Docker** to validate Thor (`close%`, compile, relative TRT 10.14 behavior). Use **native TRT 11** for best 5090 performance.

Thor timings: run the same script on Thor and compare stage-by-stage (expect closer to parity-docker TRT 10.14 than to native TRT 11).

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

## Performance metrics (reference)

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
