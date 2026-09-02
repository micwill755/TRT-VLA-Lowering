"""Shared example runtime: paths, plugin, dry-run compile knobs, graph dump."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch_tensorrt

_EXAMPLES_DIR = Path(__file__).resolve().parent
GRAPH_DIR = _EXAMPLES_DIR.parent
TEST_ROOT = GRAPH_DIR.parent
for _path in (TEST_ROOT, GRAPH_DIR):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from exporter import EdgeConfig, dump_graph, named_runtime_preview

logging.getLogger("torch._library.fake_class_registry").setLevel(logging.ERROR)

_PLUGIN_CANDIDATES = (
    Path("/home/micwilliams/workspace/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
    TEST_ROOT.parent / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so",
    Path("/home/micwilliams/workspace/TensorRT-Edge-LLM/build-plugin-trt10/libNvInfer_edgellm_plugin.so"),
    TEST_ROOT.parent / "TensorRT-Edge-LLM" / "build-plugin-trt10" / "libNvInfer_edgellm_plugin.so",
)


def ensure_plugin_so() -> None:
    if os.environ.get("EDGE_LLM_PLUGIN_SO") or os.environ.get("EDGELLM_PLUGIN_PATH"):
        return
    for candidate in _PLUGIN_CANDIDATES:
        if candidate.is_file():
            os.environ["EDGE_LLM_PLUGIN_SO"] = str(candidate)
            return
    raise RuntimeError("Set EDGE_LLM_PLUGIN_SO to libNvInfer_edgellm_plugin.so")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Build TRT engines (default is dry-run partition only)",
    )
    parser.add_argument(
        "--retrace",
        action="store_true",
        help="Retrace after --compile (screenshot execute_engine IR)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Serialize named engines (vision / language / action*) to this directory",
    )
    return parser.parse_args()


def edge_config(*, compile_engines: bool, full: bool = False) -> EdgeConfig:
    return EdgeConfig(
        require_full_compilation=full,
        decompose_attention=False,
        compare=compile_engines and not full,
        extra_compile_kwargs={
            "disable_tf32": True,
            "use_fp32_acc": True,
            "use_explicit_typing": True,
            "truncate_double": True,
            "assume_dynamic_shape_support": True,
            "offload_module_to_cpu": False,
            "dryrun": not compile_engines,
        },
    )


def dump_partition(compiled: torch.fx.GraphModule, title: str) -> None:
    dump_graph(compiled, title)
    n_acc = sum(
        1 for n in compiled.graph.nodes if n.op == "call_module" and "_run_on_acc" in str(n.target)
    )
    n_gpu = sum(
        1 for n in compiled.graph.nodes if n.op == "call_module" and "_run_on_gpu" in str(n.target)
    )
    print(f"\nTRT islands={n_acc}  leftover GPU islands={n_gpu}")
    leftover = getattr(compiled, "_run_on_gpu_1", None)
    if leftover is not None:
        dump_graph(leftover, "leftover  _run_on_gpu_1")


def maybe_retrace(
    compiled,
    step_args: tuple,
    *,
    compile_engines: bool,
    retrace: bool,
    policy=None,
    components: tuple[str, ...] = ("vision", "language"),
) -> None:
    if not compile_engines:
        print("\ndry-run only (no engines). Re-run with --compile to build TRT.")
        return
    if not retrace:
        return
    retraced = torch_tensorrt.dynamo.export(compiled, arg_inputs=step_args)
    dump_graph(retraced.graph_module, "retrace")
    if policy is not None:
        from exporter import EdgeExporter

        EdgeExporter().specialize(retraced.graph_module, policy, *components)
        dump_graph(retraced.graph_module, "graph 3 (edgellm ops)")
    else:
        named_runtime_preview(retraced.graph_module)
