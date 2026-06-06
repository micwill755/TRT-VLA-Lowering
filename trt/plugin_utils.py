import os
from typing import Tuple

import tensorrt as trt
import torch

def _register_plugin_op_impl() -> None:
    """
    Internal implementation to register the tensorrt_edge_llm::xqa_attn custom op for PyTorch.

    The Python tracing op accepts 5 tensor inputs:
    - qkv: [B, S, (Hq+Hk+Hv)*D] fused QKV tensor
    - kv: [B, 2, Hkv, Capacity, D] KV cache tensor
    - ctx_len: [B] context length per batch
    - rope: [1, MaxSeqLen, RotaryDim] rotary position encoding
    - kv_cache_start_idx: [B] starting index in KV cache

    The TensorRT converter slices qkv into the separate q, k, and v tensors
    required by the C++ AttentionPlugin. Output KV shape matches the full KV-cache input: [B, 2, Hkv, Capacity, D].
    """

    @torch.library.custom_op("tensorrt_edge_llm::xqa_attn", mutates_args=())
    def attn(
        qkv: torch.Tensor,
        kv: torch.Tensor,
        ctx_len: torch.Tensor,
        rope: torch.Tensor,
        kv_cache_start_idx: torch.Tensor,
        nq: int,
        nkv: int,
        d: int,
        enable_bidirectional_prefill: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        attn_out = torch.zeros(
            batch_size, seq_len, nq, d, dtype=qkv.dtype, device=qkv.device
        )
        updated_kv = torch.zeros_like(kv)
        return attn_out, updated_kv

    @torch.library.register_fake("tensorrt_edge_llm::xqa_attn")
    def _(qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d, enable_bidirectional_prefill):
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        attn_out = torch.empty(
            batch_size, seq_len, nq, d, dtype=qkv.dtype, device=qkv.device
        )
        updated_kv = torch.empty_like(kv)
        return attn_out, updated_kv

def register_plugin_op() -> None:
    """
    Register the tensorrt_edge_llm::xqa_attn custom op for PyTorch.

    This function is idempotent - safe to call multiple times.
    """
    if hasattr(torch.ops, "tensorrt_edge_llm") and hasattr(
        torch.ops.tensorrt_edge_llm, "xqa_attn"
    ):
        return
    _register_plugin_op_impl()

register_plugin_op()

import torch_tensorrt.dynamo.conversion.edge_plugins as edge_plugins
from trt import plugin_converter as _plugin_converter  # noqa: F401

def load_plugin():
    return load_edge_vit_attention_plugin()


def load_edge_vit_attention_plugin():
    plugin_so = os.environ.get("EDGE_LLM_PLUGIN_SO") or os.environ.get("EDGELLM_TRT_PLUGIN_SO")
    if not plugin_so:
        raise RuntimeError("Set EDGE_LLM_PLUGIN_SO to libNvInfer_edgellm_plugin.so before running this script")

    edge_plugins.load_edge_plugin(plugin_so)
    trt.init_libnvinfer_plugins(None, "")
    return plugin_so


def restore_attention(patched):
    for layer, original_attn in patched:
        layer.self_attn = original_attn


def patch_vision_attention(
    vision_model,
    *,
    batch_size: int,
    seq_len: int,
    name: str,
    allow_attention_mask: bool = False,
):
    from trt.attention import ViTPluginAttention

    patched = []

    for layer in vision_model.encoder.layers:
        patched.append((layer, layer.self_attn))
        layer.self_attn = ViTPluginAttention(
            layer.self_attn,
            batch_size=batch_size,
            seq_len=seq_len,
            name=name,
            allow_attention_mask=allow_attention_mask,
        ).eval()

    print(f"patched {name} attention modules: {len(patched)}")
    return patched


@torch.no_grad()
def infer_siglip_seq_len(vision_model, image):
    hidden_states = vision_model.embeddings(image)
    return int(hidden_states.shape[0]), int(hidden_states.shape[1])

@torch.no_grad()
def infer_smolvlm_seq_len(vision_model, image):
    patch_size = vision_model.patch_size
    patch_attention_mask = torch.ones(
        image.shape[0],
        image.shape[2] // patch_size,
        image.shape[3] // patch_size,
        dtype=torch.bool,
        device=image.device,
    )
    hidden_states = vision_model.embeddings(
        pixel_values=image,
        patch_attention_mask=patch_attention_mask,
    )
    return int(hidden_states.shape[0]), int(hidden_states.shape[1])
