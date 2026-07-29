"""TensorRT-native RoPE and attention ops for Cosmos policy export.

The eager implementations are numerically real so A/C parity remains useful.
The registered Torch-TensorRT converters lower the same calls to TensorRT's
native ``IRotaryEmbeddingLayer`` and ``IAttention`` layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import tensorrt as trt
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor


@torch.library.custom_op("wfm::cosmos_rope", mutates_args=())
def cosmos_rope(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply non-interleaved RoPE to ``x`` for eager/reference execution."""
    cos = cos_cache[position_ids.to(torch.int64)]
    sin = sin_cache[position_ids.to(torch.int64)]
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1).to(x.dtype)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1).to(x.dtype)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


@cosmos_rope.register_fake
def _cosmos_rope_fake(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    del cos_cache, sin_cache, position_ids
    return torch.empty_like(x)


def cosmos_rope_packed(
    x: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Adapt the Cosmos3Runtime packed RoPE binding to native TRT RoPE."""
    half = x.shape[-1] // 2
    cos = rope_rotary_cos_sin[..., :half].reshape(-1, half).to(x.dtype)
    sin = rope_rotary_cos_sin[..., half:].reshape(-1, half).to(x.dtype)
    return cosmos_rope(x, cos, sin, position_ids)


@torch.library.custom_op("wfm::cosmos_attention", mutates_args=())
def cosmos_attention(
    scaled_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """Run attention with an already-scaled query for eager execution."""
    return F.scaled_dot_product_attention(
        scaled_query,
        key,
        value,
        is_causal=is_causal,
        scale=1.0,
        enable_gqa=True,
    )


@cosmos_attention.register_fake
def _cosmos_attention_fake(
    scaled_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    del key, value, is_causal
    return torch.empty_like(scaled_query)


@dynamo_tensorrt_converter(
    torch.ops.wfm.cosmos_rope.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_cosmos_rope(
    ctx: ConversionContext,
    target,
    args,
    kwargs,
    name: str,
):
    del target, kwargs
    x, cos, sin, position_ids = (
        get_trt_tensor(ctx, arg, f"{name}_input_{index}")
        for index, arg in enumerate(args[:4])
    )
    head_dim = int(x.shape[-1])
    layer = ctx.net.add_rotary_embedding(
        x,
        cos,
        sin,
        False,
        head_dim,
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create Cosmos RotaryEmbedding")
    layer.name = name
    layer.set_input(3, position_ids)
    return layer.get_output(0)


@dynamo_tensorrt_converter(
    torch.ops.wfm.cosmos_attention.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_cosmos_attention(
    ctx: ConversionContext,
    target,
    args,
    kwargs,
    name: str,
):
    del target, kwargs
    query, key, value = (
        get_trt_tensor(ctx, arg, f"{name}_input_{index}")
        for index, arg in enumerate(args[:3])
    )
    layer = ctx.net.add_attention(
        query,
        key,
        value,
        trt.AttentionNormalizationOp.SOFTMAX,
        bool(args[3]),
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create Cosmos Attention")
    layer.name = name
    layer.decomposable = True
    return layer.get_output(0)
