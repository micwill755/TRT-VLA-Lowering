from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from trt.prefix_cache import PrefixKVCache


def build_qwen3vl_causal_mask(
    batch_size: int,
    q_len: int,
    prefix_len: int,
    attention_mask: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    neg_inf = torch.finfo(torch.float32).min
    kv_len = prefix_len + q_len

    if q_len == 1:
        causal_mask = torch.zeros(batch_size, 1, 1, kv_len, device=device, dtype=torch.float32)
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                keep = attention_mask[:, :, -1:, :].to(torch.bool)
            else:
                keep = attention_mask[:, None, None, :].to(torch.bool)
            causal_mask = torch.where(
                keep,
                causal_mask,
                torch.full((), neg_inf, dtype=torch.float32, device=device),
            )
        return causal_mask.to(dtype=dtype)

    future = torch.triu(torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1)
    causal_q = torch.zeros(q_len, q_len, device=device, dtype=torch.float32).masked_fill(
        future, neg_inf
    )
    base = torch.cat(
        [
            torch.zeros(q_len, prefix_len, device=device, dtype=torch.float32),
            causal_q,
        ],
        dim=-1,
    )
    causal_mask = base.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len)
    if attention_mask is not None:
        keep = attention_mask[:, None, None, :].to(torch.bool)
        causal_mask = causal_mask.masked_fill(~keep, neg_inf)
    return causal_mask.to(dtype=dtype)


class Qwen3VLTextModelPrefillExportModule(nn.Module):
    """Alpamayo VLM prefill export: fused text+vision embeddings -> prefix KV."""

    def __init__(
        self,
        language_model: nn.Module,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.language_model = language_model
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)

    @staticmethod
    def _apply_dense_deepstack(
        hidden_states: torch.Tensor,
        deepstack_layer_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if deepstack_layer_embeds is None:
            return hidden_states
        return hidden_states + deepstack_layer_embeds.to(
            hidden_states.device, hidden_states.dtype
        )

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ):
        language_model = self.language_model
        bsz = inputs_embeds.shape[0]
        cache = PrefixKVCache.empty(
            num_layers=self.num_layers,
            batch_size=bsz,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        text_position_ids = position_ids[0]

        q_len = inputs_embeds.shape[1]
        if (
            attention_mask is not None
            and attention_mask.ndim == 4
            and attention_mask.shape[-2] == q_len
        ):
            expected_kv = cache.get_seq_length() + q_len
            if attention_mask.shape[-1] != expected_kv:
                if attention_mask.shape[-1] > expected_kv:
                    attention_mask = attention_mask[..., -expected_kv:]
                else:
                    pad = torch.zeros(
                        attention_mask.shape[0],
                        attention_mask.shape[1],
                        attention_mask.shape[2],
                        expected_kv - attention_mask.shape[-1],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([pad, attention_mask], dim=-1)
            causal_mask = attention_mask.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
        else:
            causal_mask = build_qwen3vl_causal_mask(
                batch_size=bsz,
                q_len=q_len,
                prefix_len=0,
                attention_mask=attention_mask,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        if isinstance(deepstack_visual_embeds, (list, tuple)) and len(deepstack_visual_embeds) > 0:
            deepstack_visual_embeds = torch.stack(
                [
                    d.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    for d in deepstack_visual_embeds
                ],
                dim=0,
            )

        hidden_states = inputs_embeds
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            layer_embed = None
            if isinstance(deepstack_visual_embeds, torch.Tensor):
                if (
                    deepstack_visual_embeds.ndim == 4
                    and layer_idx < deepstack_visual_embeds.shape[0]
                ):
                    layer_embed = deepstack_visual_embeds[layer_idx]
                elif (
                    deepstack_visual_embeds.ndim == 3
                    and visual_pos_masks is not None
                    and layer_idx < deepstack_visual_embeds.shape[0]
                ):
                    mask3d = (
                        visual_pos_masks.to(device=inputs_embeds.device, dtype=torch.bool)
                        .unsqueeze(-1)
                        .expand(bsz, q_len, inputs_embeds.shape[-1])
                    )
                    layer_embed = torch.zeros_like(hidden_states).masked_scatter(
                        mask3d, deepstack_visual_embeds[layer_idx].reshape(-1)
                    )
            hidden_states = self._apply_dense_deepstack(hidden_states, layer_embed)

        hidden_states = language_model.norm(hidden_states)
        updated_k, updated_v = cache.get_updated_stacked()
        return hidden_states, updated_k, updated_v


def run_vlm_preprocessing(
    model: nn.Module,
    model_inputs: dict[str, Any],
    trt_vision: nn.Module | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fuse Alpamayo VLM text tokens with vision features and compute RoPE."""
    device = torch.device(device)
    tokenized_data = copy.deepcopy(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )

    vlm_model = model.vlm.model
    lm_ref = vlm_model.language_model

    original_fwd = None
    if trt_vision is not None:
        original_fwd = vlm_model.visual.forward
        vlm_model.visual.forward = trt_vision.forward

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        inputs_embeds = lm_ref.embed_tokens(input_ids)
        pv = tokenized_data["pixel_values"].to(device)
        igt = tokenized_data["image_grid_thw"].to(device)
        _vision_out = vlm_model.get_image_features(pv, igt)
        if isinstance(_vision_out, tuple):
            image_embeds, ds_embeds = _vision_out
        else:
            image_embeds = _vision_out.pooler_output
            ds_embeds = _vision_out.deepstack_features
        image_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vlm_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_cat
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_cat)
        vis_masks = image_mask[..., 0]
        attn = tokenized_data.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        try:
            position_ids, rope_deltas = vlm_model.get_rope_index(
                input_ids, igt, video_grid_thw=None, attention_mask=attn
            )
        except (TypeError, IndexError):
            image_token_id = vlm_model.config.image_token_id
            mm_token_type_ids = (input_ids == image_token_id).int()
            position_ids, rope_deltas = vlm_model.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=igt,
                video_grid_thw=None,
                attention_mask=attn,
            )

    if original_fwd is not None:
        vlm_model.visual.forward = original_fwd

    del pv, igt, image_embeds, image_cat, image_mask
    torch.cuda.empty_cache()
    return input_ids, inputs_embeds, ds_embeds, vis_masks, position_ids, rope_deltas
