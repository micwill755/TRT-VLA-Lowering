"""Dynamo converter: ``Fp16MoePlugin`` custom op → TRT ``Fp16MoePlugin``."""

from __future__ import annotations

import torch
from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority

from plugins.common import add_plugin_layer, as_trt_tensors, create_plugin, int_field


@dynamo_tensorrt_converter(
    torch.ops.trt_edgellm.Fp16MoePlugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_fp16_moe(ctx, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    router, hidden, fc1, fc2 = args[:4]
    num_experts = int(args[4])
    top_k = int(args[5])
    hidden_size = int(args[6])
    moe_inter_size = int(args[7])
    activation_type = int(args[8])
    norm_topk_prob = int(args[9])
    max_routed_rows = int(args[10])

    plugin = create_plugin(
        "Fp16MoePlugin",
        name,
        [
            int_field("num_experts", num_experts),
            int_field("top_k", top_k),
            int_field("hidden_size", hidden_size),
            int_field("moe_inter_size", moe_inter_size),
            int_field("activation_type", activation_type),
            int_field("norm_topk_prob", norm_topk_prob),
            int_field("max_routed_rows", max_routed_rows),
        ],
    )
    inputs = as_trt_tensors(ctx, [router, hidden, fc1, fc2], name)
    layer = add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0)
