"""Shared Torch-TRT plugin helpers for the example export flows.

Mirrors ``trt/plugin/plugin_converter.py``: look up an Edge-LLM V3 creator,
pack ``PluginField``s, and insert the layer into the Dynamo network.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import tensorrt as trt
import torch
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

from trt.plugin.plugin_utils import get_trt_plugin_creator, load_plugin


def int_field(name: str, value: int) -> trt.PluginField:
    return trt.PluginField(
        name,
        np.array([int(value)], dtype=np.int32),
        trt.PluginFieldType.INT32,
    )


def float_field(name: str, value: float) -> trt.PluginField:
    return trt.PluginField(
        name,
        np.array([float(value)], dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )


def _creator_is_v3(creator) -> bool:
    return "V3" in type(creator).__name__


def _plugin_is_v3(plugin) -> bool:
    return "V3" in type(plugin).__name__


def create_plugin(plugin_name: str, layer_name: str, field_list: list, version: str = "1"):
    creator = get_trt_plugin_creator(plugin_name, version, "")
    if creator is None:
        raise RuntimeError(
            f"{plugin_name} not found in the TensorRT plugin registry. "
            "Set EDGE_LLM_PLUGIN_SO / EDGELLM_PLUGIN_PATH to libNvInfer_edgellm_plugin.so"
        )
    fields = trt.PluginFieldCollection(field_list)
    if _creator_is_v3(creator):
        plugin = creator.create_plugin(layer_name, fields, trt.TensorRTPhase.BUILD)
    else:
        plugin = creator.create_plugin(layer_name, fields)
    if plugin is None:
        raise RuntimeError(f"Failed to create {plugin_name}")
    return plugin


def add_plugin_layer(ctx, inputs: list, plugin, name: str):
    layer = (
        ctx.net.add_plugin_v3(inputs, [], plugin)
        if _plugin_is_v3(plugin)
        else ctx.net.add_plugin_v2(inputs, plugin)
    )
    layer.name = name
    return layer


def as_trt_tensors(ctx, tensors: list, name: str) -> list:
    out = []
    for idx, tensor in enumerate(tensors):
        tensor_name = f"{name}_i{idx}"
        trt_tensor = (
            get_trt_tensor(ctx, tensor, tensor_name)
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        if not getattr(trt_tensor, "name", None):
            trt_tensor.name = tensor_name
        out.append(trt_tensor)
    return out


def has_torch_op(namespace: str, name: str) -> bool:
    return hasattr(torch.ops, namespace) and hasattr(getattr(torch.ops, namespace), name)


def load_example_plugins(*, include_attention: bool = False, load_so: bool = True) -> str | None:
    """Register example custom ops, optionally load the Edge-LLM .so, then import converters.

    Same order as ``trt.plugin.plugin_utils.load_plugins_for_trt``. Use
    ``load_so=False`` for export-only runs that inspect the FX graph.
    """
    from plugins.moe import ops as moe_ops
    from plugins.spec_decode import ops as spec_ops
    from plugins.ssm import ops as ssm_ops

    ssm_ops.register_ops()
    moe_ops.register_ops()
    spec_ops.register_ops()

    if include_attention:
        from trt.plugin.plugin_utils import (
            _register_attention_plugin_op,
            _register_vit_attention_plugin_op,
        )

        _register_attention_plugin_op()
        _register_vit_attention_plugin_op()

    plugin_so = None
    if load_so:
        if include_attention:
            from trt.plugin.plugin_utils import load_plugins_for_trt

            load_plugins_for_trt()
        else:
            plugin_so = load_plugin()
        from plugins.moe import converter as _moe_converter  # noqa: F401
        from plugins.spec_decode import converter as _spec_converter  # noqa: F401
        from plugins.ssm import converter as _ssm_converter  # noqa: F401
        plugin_so = plugin_so or (
            os.environ.get("EDGE_LLM_PLUGIN_SO")
            or os.environ.get("EDGELLM_TRT_PLUGIN_SO")
            or os.environ.get("EDGELLM_PLUGIN_PATH")
        )

    return plugin_so


def export_and_maybe_compile(module, args, *, label: str, compile_engine: bool):
    """Export the patched module and optionally lower custom ops through converters."""
    import torch_tensorrt

    from trt.compile import export_trt_module, make_input_spec

    module = module.eval()
    exported = export_trt_module(module, args)
    print_custom_ops(exported, label=label)
    if not compile_engine:
        print(f"{label}: skipped TRT compile")
        return exported, None
    print(f"{label}: compiling TensorRT engine (plugin converters must be registered)")
    engine = torch_tensorrt.dynamo.compile(
        exported,
        inputs=make_input_spec(args),
        skip_decompositions=True,
        **TRT_SETTINGS,
    )
    return exported, engine


def print_custom_ops(exported: Any, *, label: str) -> None:
    print(f"\n=== {label}: custom ops in exported graph ===")
    found = False
    for node in exported.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if "trt" in target or "plugin" in target:
            print(f"  {node.name}: {target}")
            found = True
    if not found:
        print("  (none — the graph never called a plugin custom op)")


TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "require_full_compilation": True,
}
