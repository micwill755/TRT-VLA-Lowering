# Torch-TRT Pipelines

Export, inference, and benchmark orchestration for VLA models on top of [TensorRT Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/).

The CLI entry point is `app.py`. It builds an `EdgeContext` from a model profile and sample dataset frame, then runs the export and/or benchmark pipelines.

## Prerequisites

- Python 3.10+ with PyTorch, Torch-TensorRT, and LeRobot dependencies (e.g. `lerobot` conda env)
- CUDA GPU
- Built Edge-LLM TensorRT plugin:

```bash
export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
```

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
