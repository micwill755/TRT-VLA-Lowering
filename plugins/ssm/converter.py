"""Dynamo converters: SSM custom ops → Edge-LLM TensorRT plugins."""

from __future__ import annotations

import torch
from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority

from plugins.common import add_plugin_layer, as_trt_tensors, create_plugin, int_field


@dynamo_tensorrt_converter(
    torch.ops.trt_edgellm.causal_conv1d.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_causal_conv1d(ctx, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    hidden, weight, bias, conv_state, context_lengths = args[:5]
    stride = int(args[5])
    padding = int(args[6])
    dilation = int(args[7])
    groups = int(args[8])

    plugin = create_plugin(
        "causal_conv1d",
        name,
        [
            int_field("stride", stride),
            int_field("padding", padding),
            int_field("dilation", dilation),
            int_field("groups", groups),
            int_field("use_mtp", 0),
            int_field("use_ddtree", 0),
        ],
    )
    inputs = as_trt_tensors(
        ctx,
        [hidden, weight, bias, conv_state, context_lengths],
        name,
    )
    layer = add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt_edgellm.update_ssm_state.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_update_ssm_state(ctx, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    hidden, ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, state, context_lengths = args[:9]
    dt_softplus = int(args[9])
    ngroups = int(args[10])
    nheads = int(args[11])
    head_dim = int(args[12])
    dstate = int(args[13])

    plugin = create_plugin(
        "update_ssm_state",
        name,
        [
            int_field("dim", head_dim),
            int_field("dstate", dstate),
            int_field("nheads", nheads),
            int_field("ngroups", ngroups),
            int_field("dt_softplus", dt_softplus),
        ],
    )
    inputs = as_trt_tensors(
        ctx,
        [hidden, ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, state, context_lengths],
        name,
    )
    layer = add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt_edgellm.gated_delta_net.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_gated_delta_net(ctx, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, a, b, a_log, dt_bias, h0, context_lengths = args[:9]
    k_dim = int(args[9])
    v_dim = int(args[10])

    plugin = create_plugin(
        "gated_delta_net",
        name,
        [
            int_field("k_dim", k_dim),
            int_field("v_dim", v_dim),
            int_field("use_mtp", 0),
            int_field("use_ddtree", 0),
        ],
    )
    inputs = as_trt_tensors(
        ctx,
        [q, k, v, a, b, a_log, dt_bias, h0, context_lengths],
        name,
    )
    layer = add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)
