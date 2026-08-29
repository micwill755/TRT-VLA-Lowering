"""Custom op for DFlash target KV-cache update (speculative decoding).

Tree attention itself is still ``AttentionPlugin`` with ``enable_tree_attention=1``.
The only *new* TRT plugin on the DFlash draft path is this KV write.
"""

from __future__ import annotations

import torch

from plugins.common import has_torch_op

# Must match ``rt::kTOKENS_PER_PAGE`` / ``tensorrt_edgellm.models.ops.KV_PAGE_SIZE``.
KV_PAGE_SIZE = 128


def register_ops() -> None:
    if has_torch_op("trt_edgellm", "dflash_target_kv_cache_update"):
        return

    @torch.library.custom_op("trt_edgellm::dflash_target_kv_cache_update", mutates_args=())
    def dflash_target_kv_cache_update(
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_cos_sin: torch.Tensor,
        delta_start_positions: torch.Tensor,
        delta_lengths: torch.Tensor,
        pages_per_slot: int,
    ) -> torch.Tensor:
        del k_delta, v_delta, rope_cos_sin, delta_start_positions, delta_lengths
        del pages_per_slot
        return past_key_value.clone()

    @dflash_target_kv_cache_update.register_fake
    def _(
        k_delta,
        v_delta,
        past_key_value,
        rope_cos_sin,
        delta_start_positions,
        delta_lengths,
        pages_per_slot,
    ):
        del k_delta, v_delta, rope_cos_sin, delta_start_positions, delta_lengths
        del pages_per_slot
        return torch.empty_like(past_key_value)
