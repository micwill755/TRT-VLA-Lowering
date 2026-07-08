from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_orig_qwen3vl_attn_forward = None


def patch_qwen3vl_vision_attention() -> bool:
    """Patch Qwen3-VL vision attention for static-length SDPA export."""
    global _orig_qwen3vl_attn_forward

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
    except ImportError:
        logger.warning("Could not import Qwen3VL attention modules — patch not applied")
        return False

    if _orig_qwen3vl_attn_forward is not None:
        return True

    _orig_qwen3vl_attn_forward = Qwen3VLVisionAttention.forward

    def _static_lengths_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        static_lengths = getattr(self, "_static_lengths", None)
        if static_lengths is None or self.config._attn_implementation == "flash_attention_2":
            return _orig_qwen3vl_attn_forward(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        splits = [
            torch.split(t, static_lengths, dim=2)
            for t in (query_states, key_states, value_states)
        ]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits, strict=True)
        ]

        attn_output = torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    Qwen3VLVisionAttention.forward = _static_lengths_forward
    logger.info("Patched Qwen3VLVisionAttention.forward for static lengths")
    return True


class VisualFixedGrid(nn.Module):
    """Qwen3-VL vision wrapper with grid_thw baked into static buffers."""

    def __init__(self, visual: nn.Module, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual.eval()

        with torch.no_grad():
            pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)

            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
            seq_len = pos_embeds.shape[0]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            static_lengths = (
                torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
                .cpu()
                .tolist()
            )
            static_lengths = [int(x) for x in static_lengths]
            for blk in self.visual.blocks:
                blk.attn._static_lengths = static_lengths

        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del grid_thw
        hidden_states = self.visual.patch_embed(hidden_states)
        torch._check(hidden_states.shape[0] != 0)
        hidden_states = hidden_states + self.pos_embeds.to(hidden_states.dtype)

        position_embeddings = (
            self.cos.to(hidden_states.dtype),
            self.sin.to(hidden_states.dtype),
        )

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                idx = self.visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.visual.deepstack_merger_list[idx](hidden_states)
                )

        hidden_states = self.visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists
