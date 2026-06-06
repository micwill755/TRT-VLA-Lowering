"""
TensorRT converter for Edge-LLM attention plugin ops.

This module contains the TensorRT converter for the tensorrt_edge_llm::xqa_attn
custom op. It is kept in a separate file from plugin_utils.py for maintainability.
"""

import numpy as np
import tensorrt as trt
from trt.plugin_utils import register_plugin_op
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

# Ensure the custom op is registered before the converter decorator runs
register_plugin_op()

import torch  # noqa: E402 (must be after register_plugin_op so the op exists)

def _slice_qkv_for_attention_plugin(
    ctx: ConversionContext,
    qkv: trt.ITensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    name: str,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    q_width = int(num_q_heads) * int(head_size)
    kv_width = int(num_kv_heads) * int(head_size)
    batch_dim = qkv.shape[0]
    seq_dim = qkv.shape[1]

    q_layer = ctx.net.add_slice(qkv, (0, 0, 0), (batch_dim, seq_dim, q_width), (1, 1, 1))
    q_layer.name = f"{name}_q_slice"
    q = q_layer.get_output(0)

    k_layer = ctx.net.add_slice(qkv, (0, 0, q_width), (batch_dim, seq_dim, kv_width), (1, 1, 1))
    k_layer.name = f"{name}_k_slice"
    k = k_layer.get_output(0)

    v_layer = ctx.net.add_slice(qkv, (0, 0, q_width + kv_width), (batch_dim, seq_dim, kv_width), (1, 1, 1))
    v_layer.name = f"{name}_v_slice"
    v = v_layer.get_output(0)

    return q, k, v


@dynamo_tensorrt_converter(
    torch.ops.tensorrt_edge_llm.xqa_attn.default, supports_dynamic_shapes=True
)
def convert_attn(ctx: ConversionContext, target, args, kwargs, name):
    """
    Convert tensorrt_edge_llm::xqa_attn op to TensorRT AttentionPlugin.

    The Python op takes fused qkv for tracing compatibility. The current C++
    AttentionPlugin expects separate q, k, and v tensors, so this converter
    slices qkv before creating the plugin layer.
    """
    del target, kwargs

    qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d, enable_bidirectional_prefill = list(args)[:9]
    nq = int(nq)
    nkv = int(nkv)
    d = int(d)
    enable_bidirectional_prefill = int(enable_bidirectional_prefill)

    creator = trt.get_plugin_registry().get_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            field_name, np.array([field_val], dtype=np.int32), trt.PluginFieldType.INT32
        )
        for field_name, field_val in [
            ("num_q_heads", nq),
            ("num_kv_heads", nkv),
            ("head_size", d),
            ("enable_tree_attention", 0),
            ("enable_fp8_kv_cache", 0),
            ("sliding_window_size", -1),
            ("enable_bidirectional_prefill", enable_bidirectional_prefill),
        ]
    ]

    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create TensorRT AttentionPlugin")

    qkv = get_trt_tensor(ctx, qkv, f"{name}_qkv") if not isinstance(qkv, trt.ITensor) else qkv
    q, k, v = _slice_qkv_for_attention_plugin(ctx, qkv, nq, nkv, d, name)
    plugin_inputs = [q, k, v, kv, ctx_len, rope, kv_cache_start_idx]
    inputs = [
        (
            get_trt_tensor(ctx, tensor, f"{name}_i{idx}")
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        for idx, tensor in enumerate(plugin_inputs)
    ]

    if len(inputs[4].shape) == 2 and inputs[4].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[4])
        shuffle_layer.reshape_dims = (inputs[4].shape[0],)
        inputs[4] = shuffle_layer.get_output(0)

    if len(inputs[6].shape) == 2 and inputs[6].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[6])
        shuffle_layer.reshape_dims = (inputs[6].shape[0],)
        inputs[6] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    layer.name = name
    return layer.get_output(0), layer.get_output(1)
