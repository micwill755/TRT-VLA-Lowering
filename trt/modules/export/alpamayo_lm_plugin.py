# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM plugin-based LM compilation for Alpamayo's VLM language model.

Port of ``alpamayo_r1.trt.lm_plugin`` into the Test project, adapted to Test's
``trt::attention_plugin`` / ``PluginAttention`` stack (RoPE + kvcache start
index are graph/runtime inputs rather than Alpamayo's baked ``xqa_attn`` path).

Compiled contract::

    (inputs_embeds, kv_caches, ctx_len, ds_stack) -> (logits, updated_kv_caches)
"""

from __future__ import annotations

import copy
import gc
import logging
from contextlib import nullcontext
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from trt.plugin.attention import ContextAttentionMaskType, PluginAttention
from trt.plugin.plugin_utils import (
    create_kv_caches,
    load_plugins_for_trt,
    set_plugin_config_from_model,
)

logger = logging.getLogger(__name__)

FP16 = torch.float16


class PluginWrapperDSInput(nn.Module):
    """LM forward with plugin self-attention and per-layer deepstack adds."""

    def __init__(self, lm: nn.Module, lm_head: nn.Module, num_ds: int, rope_cache: torch.Tensor):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.num_ds = int(num_ds)
        # RoPE is an AttentionPlugin input in Test; keep it as a buffer so the
        # wrapper I/O stays (embeds, kvs, ctx, ds) like Alpamayo's lm_plugin.
        self.register_buffer("rope_cache", rope_cache.float(), persistent=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden = inputs_embeds
        # Keep symbolic shapes (do not int()-cast) so torch.export can treat
        # seq_len as dynamic, matching Alpamayo's PluginWrapperDSInput.
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        kvcache_start_index = torch.zeros(
            batch_size, dtype=torch.int32, device=inputs_embeds.device
        )
        new_kvs: list[torch.Tensor] = []
        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=self.rope_cache,
                past_key_value=kv_caches[i],
                ctx_len=ctx_len,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = residual + hidden
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            new_kvs.append(kv)
            if i < self.num_ds:
                hidden = hidden + ds_stack[i, :, :seq_len, :]
        hidden = self.lm.norm(hidden)
        return self.lm_head(hidden), new_kvs


def build_rope_cache(
    lm: nn.Module,
    S_input: int,
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-compute concatenated ``(cos, sin)`` RoPE cache up to ``max_seq_len``."""
    with torch.no_grad():
        d_eff = torch.arange(S_input, max_seq_len, device=device).float()
        d_eff = d_eff + rope_deltas.to(device).float().squeeze()
        d_3d = d_eff.view(1, 1, -1).expand(3, 1, -1).long()
        full_pos = torch.cat([position_ids.to(device), d_3d], dim=2)
        cos, sin = lm.rotary_emb(torch.ones(1, device=device, dtype=FP16), full_pos)
        h2 = head_dim // 2
        rope_cache = torch.cat(
            [cos[:, :max_seq_len, :h2].float(), sin[:, :max_seq_len, :h2].float()],
            dim=-1,
        )
    return rope_cache

def plugin_kvs_to_prefix(
    kv_caches: List[torch.Tensor], prefix_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert plugin KV list ``[B,2,nkv,cap,d]`` to stacked prefix K/V for diffusion."""
    stacked = torch.stack(kv_caches, dim=0)  # [L, B, 2, nkv, cap, d]
    prefix_k = stacked[:, :, 0, :, :prefix_len, :].contiguous()
    prefix_v = stacked[:, :, 1, :, :prefix_len, :].contiguous()
    return prefix_k, prefix_v


def pack_deepstack_to_ds_stack(
    ds_embeds: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    visual_pos_masks: torch.Tensor,
    *,
    batch_size: int,
    max_seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Scatter deepstack vision features into ``[num_ds, B, max_seq, H]`` for the plugin LM.

    Matches Alpamayo ``run_inference_trt_plugin`` packing: zeros everywhere except
    visual placeholder positions filled from each deepstack layer tensor.
    """
    if isinstance(ds_embeds, torch.Tensor):
        if ds_embeds.ndim == 4:
            layers = [ds_embeds[i] for i in range(ds_embeds.shape[0])]
        else:
            layers = [ds_embeds]
    else:
        layers = list(ds_embeds)

    num_ds = len(layers)
    ds_stack = torch.zeros(
        num_ds, batch_size, max_seq_len, hidden_size, dtype=dtype, device=device
    )
    vis_mask_row = visual_pos_masks
    while vis_mask_row.dim() > 2:
        vis_mask_row = vis_mask_row[..., 0]
    if vis_mask_row.dim() == 2:
        vp = vis_mask_row[0].nonzero(as_tuple=True)[0]
    else:
        vp = vis_mask_row.nonzero(as_tuple=True)[0]

    for i, de in enumerate(layers):
        de = de.to(device=device, dtype=dtype)
        if de.dim() > 2:
            de = de.reshape(-1, de.shape[-1])
        if de.numel() > 0 and vp.numel() > 0:
            if de.shape[0] != int(vp.numel()):
                raise ValueError(
                    f"deepstack layer {i} has {de.shape[0]} tokens but "
                    f"visual_pos_masks has {int(vp.numel())} positions"
                )
            ds_stack[i, :, vp, :] = de.unsqueeze(0).expand(batch_size, -1, -1)
    return ds_stack

