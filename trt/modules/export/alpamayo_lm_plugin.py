# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alpamayo language export helpers (deepstack + mRoPE cache for CausalLMExportModule)."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def build_alpamayo_rope_cache(
    lm: nn.Module,
    *,
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    seq_len: int,
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Bake Qwen3-VL mRoPE into AttentionPlugin ``rope_rotary_cos_sin`` layout.

    Matches ``alpamayo_r1.trt.lm_plugin._build_rope_cache``: call the model's
    ``rotary_emb`` with 3D ``position_ids`` from ``get_rope_index``, extend past
    ``seq_len`` with ``rope_deltas`` when ``max_seq_len > seq_len``, then pack
    ``[cos[:h2] | sin[:h2]]`` as FP32 ``[1, max_seq_len, head_dim]``.
    """
    seq_len = int(seq_len)
    max_seq_len = int(max_seq_len)
    head_dim = int(head_dim)
    rotary_emb = getattr(lm, "rotary_emb", None)
    if rotary_emb is None:
        rotary_emb = getattr(getattr(lm, "model", None), "rotary_emb", None)
    if rotary_emb is None:
        raise AttributeError("language model has no rotary_emb for Alpamayo mRoPE")

    d_eff = torch.arange(seq_len, max_seq_len, device=device).float()
    d_eff = d_eff + rope_deltas.to(device=device).float().squeeze()
    d_3d = d_eff.view(1, 1, -1).expand(3, 1, -1).long()
    full_pos = torch.cat([position_ids.to(device=device), d_3d], dim=2)
    cos, sin = rotary_emb(
        torch.ones(1, device=device, dtype=torch.float16),
        full_pos,
    )
    h2 = head_dim // 2
    return torch.cat(
        [cos[:, :max_seq_len, :h2].float(), sin[:, :max_seq_len, :h2].float()],
        dim=-1,
    ).contiguous()


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
    """Scatter deepstack vision features into ``[num_ds, B, max_seq, H]``."""
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
