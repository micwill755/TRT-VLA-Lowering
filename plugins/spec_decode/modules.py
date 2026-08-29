"""Spec-decode plugin wrappers.

* ``PluginDFlashKVUpdate`` — ``DFlashTargetKVCacheUpdate`` (paged KV write).
* ``PluginTreeAttention`` — existing ``AttentionPlugin`` with tree mask / pos ids,
  same converter as the VLA language path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from plugins.spec_decode.ops import KV_PAGE_SIZE
from trt.plugin.attention import ContextAttentionMaskType


class PluginDFlashKVUpdate(nn.Module):
    """Apply RoPE to target K-delta and scatter into the draft paged KV pool."""

    def __init__(self, pages_per_slot: int):
        super().__init__()
        self.pages_per_slot = int(pages_per_slot)

    def forward(
        self,
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_cos_sin: torch.Tensor,
        delta_start_positions: torch.Tensor,
        delta_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trt_edgellm.dflash_target_kv_cache_update.default(
            k_delta,
            v_delta,
            past_key_value,
            rope_cos_sin,
            delta_start_positions,
            delta_lengths,
            self.pages_per_slot,
        )


class PluginTreeAttention(nn.Module):
    """Proposal self-attention over the draft tree.

    Uses the VLA ``trt.attention_plugin`` converter (split Q/K/V, linear KV
    cache). Upstream Edge-LLM DFlash packs QKV and uses a paged pool; this
    example matches the Test VLA AttentionPlugin ABI.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).to(torch.float16)
        k = self.k_proj(hidden_states).to(torch.float16)
        v = self.v_proj(hidden_states).to(torch.float16)
        attn_out, updated_kv = torch.ops.trt.attention_plugin.default(
            q,
            k,
            v,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            self.num_q_heads,
            self.num_kv_heads,
            True,
            self.head_dim,
            False,
            -1,
            int(ContextAttentionMaskType.CAUSAL),
            attention_mask,
            position_ids,
        )
        attn_out = attn_out.reshape(batch, seq_len, self.num_q_heads * self.head_dim)
        return self.o_proj(attn_out), updated_kv


def paged_kv_shape(
    max_batch: int,
    pages_per_slot: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int]:
    num_pages = int(max_batch) * int(pages_per_slot)
    return (2, num_pages, KV_PAGE_SIZE, int(num_kv_heads), int(head_dim))
