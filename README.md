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

## Quick start (GR00T)

From this directory:

```bash
cd /home/micwilliams/workspace/Test
```

### Export + benchmark (default)

Runs export, then benchmark parity in one command:

```bash
python app.py --model gr00t --device cuda --engine-dir /tmp/groot_edge_llm
```

### Export and benchmark separately

```bash
# 1. Compile engines to disk
python app.py --model gr00t --export-only --device cuda --engine-dir /tmp/groot_edge_llm

# 2. Run eager vs in-memory vs serialized parity (requires engines from step 1)
python app.py --model gr00t --benchmark-only --device cuda --engine-dir /tmp/groot_edge_llm
```

### Eager inference only

```bash
python app.py --model gr00t --inference-only --device cuda --engine-dir /tmp/groot_edge_llm
```

## Useful flags

| Flag | Description |
|------|-------------|
| `--model` | VLA profile (`gr00t` is fully wired today) |
| `--engine-dir` | Where engines are written/read (default: `/tmp/groot_edge_llm`) |
| `--device` | `cuda` or `cpu` |
| `--dataset-id` | LeRobot dataset (default: `lerobot/libero`) |
| `--episode-index` | Dataset episode (default: `0`) |
| `--frame-index` | Frame within episode (default: `0`) |
| `--export-only` | Export engines; skip benchmark |
| `--benchmark-only` | Benchmark only; skip export |
| `--inference-only` | Single eager inference pass |

## Engine layout

After export, GR00T writes:

```text
/tmp/groot_edge_llm/
  visual/visual.engine
  language/language.engine
  action_context/context.engine
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

Open `docs/build/html/index.html` for architecture, export module examples, and customization guides.

## Project layout

```text
app.py                          CLI entry
trt/orchestrator/               EdgeOrchestrator
trt/pipelines/                  Export, inference, benchmark pipelines
trt/executor/models/groot/      GR00T stage hooks and pipelines
trt/modules/export/             Reusable export modules (vision, language, diffusion)
docs/source/                    Sphinx documentation
```
