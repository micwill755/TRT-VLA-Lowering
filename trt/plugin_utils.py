import os
import ctypes

from typing import Any, Optional, Sequence, Tuple

import tensorrt as trt
import torch

_PLUGIN_CONFIG: dict[str, Any] = {}


def get_plugin_config() -> dict[str, Any]:
    return _PLUGIN_CONFIG


def _has_torch_op(namespace: str, name: str) -> bool:
    return hasattr(torch.ops, namespace) and hasattr(getattr(torch.ops, namespace), name)


def _register_attention_plugin_op() -> None:
    """Register ``trt::attention_plugin`` for export when edge_plugins is unavailable."""
    if _has_torch_op("trt", "attention_plugin"):
        return

    @torch.library.custom_op("trt::attention_plugin", mutates_args=())
    def attention_plugin(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        enable_tree_attention: bool,
        head_size: int,
        enable_fp8_kv_cache: bool,
        sliding_window_size: int = -1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        qkv_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del k, v, context_lengths, rope_rotary_cos_sin, kvcache_start_index
        del num_kv_heads, enable_tree_attention, enable_fp8_kv_cache
        del sliding_window_size, attention_mask, position_ids, qkv_scales
        batch_size, seq_len, _ = q.shape
        attn_output = torch.zeros(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )
        return attn_output, past_key_value.clone()

    @attention_plugin.register_fake
    def _attention_plugin_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        enable_tree_attention: bool,
        head_size: int,
        enable_fp8_kv_cache: bool,
        sliding_window_size: int = -1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        qkv_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del k, v, context_lengths, rope_rotary_cos_sin, kvcache_start_index
        del num_kv_heads, enable_tree_attention, enable_fp8_kv_cache
        del sliding_window_size, attention_mask, position_ids, qkv_scales
        batch_size, seq_len, _ = q.shape
        attn_output = torch.empty(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )
        return attn_output, torch.empty_like(past_key_value)


def _register_vit_attention_plugin_op() -> None:
    if _has_torch_op("trt", "vit_attention_plugin"):
        return

    @torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
    def vit_attention_plugin(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        del k, v, cu_seqlens, max_seqlen_carrier, num_heads, head_size
        return torch.zeros_like(q)

    @vit_attention_plugin.register_fake
    def _vit_attention_plugin_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        del k, v, cu_seqlens, max_seqlen_carrier, num_heads, head_size
        return torch.empty_like(q)


def get_trt_plugin_creator(
    plugin_name: str,
    version: str = "1",
    namespace: str = "",
):
    """TRT 10: get_plugin_creator; TRT 11+: get_creator."""
    registry = trt.get_plugin_registry()
    if hasattr(registry, "get_plugin_creator"):
        return registry.get_plugin_creator(plugin_name, version, namespace)
    return registry.get_creator(plugin_name, version, namespace)


def load_plugin():
    plugin_so = os.environ.get("EDGE_LLM_PLUGIN_SO") or os.environ.get("EDGELLM_TRT_PLUGIN_SO")
    if not plugin_so:
        raise RuntimeError("Set EDGE_LLM_PLUGIN_SO to libNvInfer_edgellm_plugin.so before running this script")

    try:
        import torch_tensorrt.dynamo.conversion.edge_plugins as edge_plugins

        edge_plugins.load_edge_plugin(plugin_so)
    except ImportError:
        ctypes.CDLL(plugin_so)
    trt.init_libnvinfer_plugins(None, "")
    return plugin_so


def load_plugins_for_trt():
    _register_attention_plugin_op()
    _register_vit_attention_plugin_op()
    load_plugin()
    from trt import plugin_converter as _plugin_converter  # noqa: F401,E402


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


def set_plugin_config(
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    enable_bidirectional_prefill: int = 0,
) -> None:
    global _PLUGIN_CONFIG
    _PLUGIN_CONFIG = {
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "max_batch_size": max_batch_size,
        "enable_bidirectional_prefill": int(enable_bidirectional_prefill),
    }


def set_plugin_config_from_model(
    model_config: Any,
    max_seq_len: int = 2048,
    enable_bidirectional_prefill: int = 0,
) -> None:
    if hasattr(model_config, "head_dim") and model_config.head_dim is not None:
        head_dim = model_config.head_dim
    else:
        head_dim = model_config.hidden_size // model_config.num_attention_heads

    set_plugin_config(
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
    )
