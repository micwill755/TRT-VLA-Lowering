"""Dynamo converter: DFlash KV update → ``DFlashTargetKVCacheUpdate``.

Tree attention reuses ``trt.plugin.plugin_converter.convert_llm_attention_plugin``.
"""

from __future__ import annotations

import torch
from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority

from plugins.common import add_plugin_layer, as_trt_tensors, create_plugin, int_field


@dynamo_tensorrt_converter(
    torch.ops.trt_edgellm.dflash_target_kv_cache_update.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_dflash_kv_update(ctx, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    k_delta, v_delta, past_kv, rope, delta_start, delta_lengths = args[:6]
    pages_per_slot = int(args[6])

    plugin = create_plugin(
        "DFlashTargetKVCacheUpdate",
        name,
        [int_field("pages_per_slot", pages_per_slot)],
    )
    inputs = as_trt_tensors(
        ctx,
        [k_delta, v_delta, past_kv, rope, delta_start, delta_lengths],
        name,
    )
    layer = add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0)
