#!/usr/bin/env python3
"""Inspect a serialized TensorRT engine (bindings + optimization profiles).

Edge-LLM language engines use AttentionPlugin; set EDGE_LLM_PLUGIN_SO or pass
--plugin-so before deserializing.

Example:
  export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
  python trt/inspect_engine.py /tmp/groot_edge_llm/language/language.engine

  python trt/inspect_engine.py /tmp/groot_edge_llm/language/language.engine \\
      --bindings inputs_embeds kvcache_start_index logits
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys

import tensorrt as trt

DEFAULT_LANGUAGE_BINDINGS = (
    "inputs_embeds",
    "kvcache_start_index",
    "context_lengths",
    "last_token_ids",
    "logits",
    "context_embs",
)


def _dims_tuple(dims: trt.Dims) -> tuple[int, ...] | str:
    try:
        rank = len(dims)
    except ValueError:
        return repr(dims)
    if rank <= 0:
        return repr(dims)
    return tuple(int(dims[i]) for i in range(rank))


def _profile_min_opt_max(engine: trt.ICudaEngine, name: str, profile_index: int) -> dict[str, tuple[int, ...] | str]:
    shapes = engine.get_tensor_profile_shape(name, profile_index)
    labels = ("MIN", "OPT", "MAX")
    out: dict[str, tuple[int, ...] | str] = {}
    for i, label in enumerate(labels):
        if i < len(shapes):
            out[label] = _dims_tuple(shapes[i])
        else:
            out[label] = "?"
    return out


def _load_plugin(plugin_so: str | None) -> str | None:
    plugin_so = plugin_so or os.environ.get("EDGE_LLM_PLUGIN_SO") or os.environ.get("EDGELLM_TRT_PLUGIN_SO")
    if not plugin_so:
        return None

    try:
        import torch_tensorrt.dynamo.conversion.edge_plugins as edge_plugins

        edge_plugins.load_edge_plugin(plugin_so)
    except ImportError:
        import ctypes

        ctypes.CDLL(plugin_so)
    trt.init_libnvinfer_plugins(None, "")
    return plugin_so


def _deserialize_engine(engine_path: pathlib.Path, plugin_so: str | None) -> trt.ICudaEngine:
    loaded = _load_plugin(plugin_so)
    if loaded:
        print(f"plugin: {loaded}")
    else:
        print("plugin: (none — set EDGE_LLM_PLUGIN_SO if deserialize fails)")

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    data = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize {engine_path}. "
            "If the engine uses AttentionPlugin, pass --plugin-so or set EDGE_LLM_PLUGIN_SO."
        )
    return engine


def _tensor_io_label(engine: trt.ICudaEngine, name: str) -> str:
    mode = engine.get_tensor_mode(name)
    return "IN " if mode == trt.TensorIOMode.INPUT else "OUT"


def _print_binding_profiles(engine: trt.ICudaEngine, name: str) -> None:
    print(f"=== {name} ({_tensor_io_label(engine, name)}) ===")
    for profile_index in range(engine.num_optimization_profiles):
        shapes = _profile_min_opt_max(engine, name, profile_index)
        print(
            f"  profile {profile_index}: "
            f"MIN={shapes['MIN']}  OPT={shapes['OPT']}  MAX={shapes['MAX']}"
        )
    print()


def _decode_readiness(engine: trt.ICudaEngine) -> bool:
    if engine.num_optimization_profiles != 2:
        print(f"profiles == 2: False (got {engine.num_optimization_profiles})")
        return False

    prefill = _profile_min_opt_max(engine, "inputs_embeds", 0)
    decode = _profile_min_opt_max(engine, "inputs_embeds", 1)

    def seq_dim(profile: dict[str, tuple[int, ...] | str], key: str) -> int | None:
        shape = profile.get(key)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return None
        return int(shape[1])

    prefill_max_seq = seq_dim(prefill, "MAX")
    decode_max_seq = seq_dim(decode, "MAX")
    decode_min_seq = seq_dim(decode, "MIN")
    decode_opt_seq = seq_dim(decode, "OPT")

    print(f"profiles == 2: True")
    print(f"prefill inputs_embeds MAX: {prefill['MAX']}")
    print(f"decode  inputs_embeds MIN/OPT/MAX: {decode['MIN']} / {decode['OPT']} / {decode['MAX']}")

    decode_ready = (
        decode_min_seq == 1
        and decode_opt_seq == 1
        and decode_max_seq == 1
        and prefill_max_seq is not None
        and prefill_max_seq > 1
    )
    print(f"decode-ready (profile 1 seq == 1): {decode_ready}")

    if not decode_ready and prefill["MAX"] == decode["MAX"]:
        print("NOTE: profile 0 and 1 inputs_embeds shapes are identical — re-export with disjoint profiles.")

    return decode_ready


def inspect_engine(
    engine_path: pathlib.Path,
    *,
    plugin_so: str | None = None,
    bindings: tuple[str, ...] | None = None,
    list_all: bool = False,
) -> int:
    engine = _deserialize_engine(engine_path, plugin_so)

    mtime = dt.datetime.fromtimestamp(engine_path.stat().st_mtime)
    print(f"engine: {engine_path}")
    print(f"modified: {mtime.isoformat(sep=' ', timespec='seconds')}")
    print(f"tensorrt: {trt.__version__}")
    print(f"num_optimization_profiles: {engine.num_optimization_profiles}")
    print(f"num_io_tensors: {engine.num_io_tensors}")
    print()

    if list_all:
        print("=== ALL BINDINGS (MAX per profile) ===")
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            parts = []
            for profile_index in range(engine.num_optimization_profiles):
                shapes = _profile_min_opt_max(engine, name, profile_index)
                parts.append(f"p{profile_index}={shapes['MAX']}")
            print(f"{index:2d} {_tensor_io_label(engine, name)} {name}: {', '.join(parts)}")
        print()

    selected = bindings or DEFAULT_LANGUAGE_BINDINGS
    print("=== SELECTED BINDINGS (min / opt / max) ===")
    for name in selected:
        try:
            engine.get_tensor_mode(name)
        except Exception:
            print(f"=== {name}: NOT FOUND ===\n")
            continue
        _print_binding_profiles(engine, name)

    if "inputs_embeds" in selected:
        print("=== DECODE READINESS ===")
        _decode_readiness(engine)
        print()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a TensorRT engine's IO bindings and optimization profiles.")
    parser.add_argument("engine", type=pathlib.Path, help="Path to .engine file")
    parser.add_argument(
        "--plugin-so",
        type=pathlib.Path,
        default=None,
        help="Path to libNvInfer_edgellm_plugin.so (or set EDGE_LLM_PLUGIN_SO)",
    )
    parser.add_argument(
        "--bindings",
        nargs="*",
        default=None,
        help="Binding names to print in detail (default: common language bindings)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also print MAX shape for every binding",
    )
    args = parser.parse_args(argv)

    if not args.engine.is_file():
        print(f"error: engine not found: {args.engine}", file=sys.stderr)
        return 1

    plugin_so = str(args.plugin_so) if args.plugin_so else None
    bindings = tuple(args.bindings) if args.bindings else None
    return inspect_engine(args.engine, plugin_so=plugin_so, bindings=bindings, list_all=args.all)


if __name__ == "__main__":
    raise SystemExit(main())
