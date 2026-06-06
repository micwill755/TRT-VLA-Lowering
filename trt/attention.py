import os
from typing import Any, Optional, Tuple

import tensorrt as trt
import torch
import torch.nn as nn
from trt.plugin_utils import register_plugin_op
register_plugin_op()

class PluginAttention(nn.Module):
    """
    Model-agnostic Plugin Attention module that replaces standard attention.

    This module wraps the projection layers from the original attention module
    and uses the tensorrt_edge_llm::xqa_attn plugin op for the attention computation.

    Supports:
    - Qwen2.5, Llama: Standard attention
    - Qwen3: Attention with QK Normalization (q_norm, k_norm)
    """

    def __init__(
        self,
        original_attn: nn.Module,
        config: Any,
        layer_idx: int,
        rope_cache: torch.Tensor,
    ):
        """
        Initialize PluginAttention.

        Args:
            original_attn: The original attention module to wrap.
            config: Model configuration.
            layer_idx: Index of this layer in the model.
            rope_cache: Pre-computed RoPE cache tensor.
        """
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Qwen3 has QK Normalization
        self.q_norm = getattr(original_attn, "q_norm", None)
        self.k_norm = getattr(original_attn, "k_norm", None)

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Qwen3 has explicit head_dim that may differ from hidden_size // num_attention_heads
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads

        # For Qwen3, attention output size is num_heads * head_dim, not hidden_size
        self.attn_hidden_size = self.num_heads * self.head_dim
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.register_buffer("rope_cache", rope_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        ctx_len: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using the plugin attention.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Unused (plugin handles masking internally).
            position_ids: Position IDs (unused, plugin uses RoPE cache).
            past_key_value: KV cache tensor of shape [batch, 2, num_kv_heads, capacity, head_dim].
            ctx_len: Context length tensor for each batch item.

        Returns:
            Tuple of (output tensor, updated KV cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Qwen3: Apply QK Normalization if available
        if self.q_norm is not None:
            # Reshape for per-head normalization: [B, S, num_heads, head_dim]
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = self.q_norm(q)
            q = q.view(batch_size, seq_len, -1)

        if self.k_norm is not None:
            # Reshape for per-head normalization: [B, S, num_kv_heads, head_dim]
            k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            k = self.k_norm(k)
            k = k.view(batch_size, seq_len, -1)

        qkv = torch.cat([q, k, v], dim=-1)

        if ctx_len is None:
            ctx_len = torch.tensor(
                [seq_len], dtype=torch.int32, device=hidden_states.device
            ).expand(batch_size)

        rope_fp32 = self.rope_cache.float()

        if past_key_value is None:
            raise ValueError("past_key_value (KV cache tensor) must be provided")

        # Empty start indices signal normal prefill with no existing KV cache.
        kv_cache_start_idx = torch.empty(
            0, dtype=torch.int32, device=hidden_states.device
        )

        attn_out, updated_kv = torch.ops.tensorrt_edge_llm.xqa_attn.default(
            qkv,
            past_key_value,
            ctx_len,
            rope_fp32,
            kv_cache_start_idx,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            1,
        )

        # Use attn_hidden_size for reshape (may differ from hidden_size in Qwen3)
        attn_out = attn_out.reshape(batch_size, seq_len, self.attn_hidden_size)
        output = self.o_proj(attn_out)
        return output, updated_kv

class ViTPluginAttention(nn.Module):
    def __init__(self, attn, *, batch_size: int, seq_len: int, name: str, allow_attention_mask: bool = False):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)
        self.name = name
        self.allow_attention_mask = bool(allow_attention_mask)

        device = self.q_proj.weight.device

        cu_seqlens = torch.arange(
            0,
            (int(batch_size) + 1) * int(seq_len),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        max_seqlen_carrier = torch.zeros(
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )

        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        self.register_buffer("max_seqlen_carrier", max_seqlen_carrier, persistent=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if attention_mask is not None and not self.allow_attention_mask:
            raise RuntimeError(f"{self.name} ViT plugin path expects no vision attention_mask")

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size * seq_len, self.num_heads, self.head_dim).to(torch.float16).contiguous()
        k = k.reshape(batch_size * seq_len, self.num_heads, self.head_dim).to(torch.float16).contiguous()
        v = v.reshape(batch_size * seq_len, self.num_heads, self.head_dim).to(torch.float16).contiguous()

        attn_output = torch.ops.trt.vit_attention_plugin.default(
            q,
            k,
            v,
            self.cu_seqlens,
            self.max_seqlen_carrier,
            self.num_heads,
            self.head_dim,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = attn_output.to(dtype=self.out_proj.weight.dtype)
        attn_output = self.out_proj(attn_output)
        return attn_output, None
