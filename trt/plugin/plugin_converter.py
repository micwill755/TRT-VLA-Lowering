"""
TensorRT converter for Edge-LLM attention plugin ops.

Overrides the stock ``trt.attention_plugin`` converter to pass separate Q/K/V
tensors directly to AttentionPlugin (no fused-qkv slice) and to honor GROOT's
``enable_bidirectional_prefill`` plugin field.
"""
import numpy as np
import tensorrt as trt
import torch
from torch_tensorrt.dynamo.conversion import ConversionContext, dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

from trt.plugin.plugin_utils import get_trt_plugin_creator


@dynamo_tensorrt_converter(
    torch.ops.trt.attention_plugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_llm_attention_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, kv, ctx_len, rope, kv_cache_start_idx = args[:7]
    num_q_heads = args[7]
    num_kv_heads = args[8]
    enable_tree_attention = args[9]
    head_size = args[10]
    enable_fp8_kv_cache = args[11]
    sliding_window_size = args[12] if len(args) > 12 else -1
    enable_bidirectional_prefill = int(args[13]) if len(args) > 13 else 0
    attention_mask = args[14] if len(args) > 14 else None
    position_ids = args[15] if len(args) > 15 else None
    qkv_scales = args[16] if len(args) > 16 else None

    creator = get_trt_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            field_name,
            np.array([field_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        )
        for field_name, field_val in [
            ("num_q_heads", int(num_q_heads)),
            ("num_kv_heads", int(num_kv_heads)),
            ("head_size", int(head_size)),
            ("enable_tree_attention", int(enable_tree_attention)),
            ("enable_fp8_kv_cache", int(enable_fp8_kv_cache)),
            ("sliding_window_size", int(sliding_window_size)),
            ("enable_bidirectional_prefill", enable_bidirectional_prefill),
        ]
    ]
    if bool(enable_fp8_kv_cache) and qkv_scales is not None:
        field_list.append(
            trt.PluginField(
                "qkv_scales",
                np.array(list(qkv_scales), dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            )
        )

    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create AttentionPlugin")

    plugin_inputs = [q, k, v, kv, ctx_len, rope, kv_cache_start_idx]
    if bool(enable_tree_attention):
        plugin_inputs.extend([attention_mask, position_ids])

    inputs = [
        get_trt_tensor(ctx, tensor, f"{name}_i{idx}")
        if not isinstance(tensor, trt.ITensor)
        else tensor
        for idx, tensor in enumerate(plugin_inputs)
    ]

    kv_cache_start_idx_input_idx = 6
    if (
        len(inputs[kv_cache_start_idx_input_idx].shape) == 2
        and inputs[kv_cache_start_idx_input_idx].shape[1] == 1
    ):
        shuffle_layer = ctx.net.add_shuffle(inputs[kv_cache_start_idx_input_idx])
        shuffle_layer.reshape_dims = (inputs[kv_cache_start_idx_input_idx].shape[0],)
        inputs[kv_cache_start_idx_input_idx] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    layer.name = name
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.vit_attention_plugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_vit_attention_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, cu_seqlens, max_seqlen_carrier = args[:5]
    num_heads = args[5]
    head_size = args[6]

    creator = get_trt_plugin_creator("ViTAttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("ViTAttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            "num_heads", np.array([int(num_heads)], dtype=np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            "head_size", np.array([int(head_size)], dtype=np.int32), trt.PluginFieldType.INT32
        ),
    ]
    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create ViTAttentionPlugin")

    inputs = []
    for idx, tensor in enumerate([q, k, v, cu_seqlens, max_seqlen_carrier]):
        tensor_name = f"{name}_i{idx}"
        trt_tensor = (
            get_trt_tensor(ctx, tensor, tensor_name)
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        if not trt_tensor.name:
            trt_tensor.name = tensor_name
        inputs.append(trt_tensor)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    layer.name = name
    output = layer.get_output(0)
    if not output.name:
        output.name = f"{name}_output"
    return output
