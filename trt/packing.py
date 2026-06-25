"""Per-VLA multimodal prefix packing for Edge-LLM export.

Each ``pack_*`` function returns a plain dict with keys:

- ``inputs_embeds``  [B, L, H]
- ``pad_mask``       [B, L]
- ``attention_mask`` [B, L, L] or [B, 1, L, L] (PI/Smol) or [B, L] (GR00T)
- ``position_ids``   [B, L]

Optional: ``image_token_mask``, ``token_type_ids`` (PI0.5 compact path).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks as pi05_make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks as smolvla_make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import pad_tensor


def _normalize_pi05_image_embs(
    image_embs: list[torch.Tensor],
    img_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Restore [B, S, H] from VitRunner-style flat [B * S, H]."""
    normalized: list[torch.Tensor] = []
    for embed, mask in zip(image_embs, img_masks, strict=True):
        if embed.ndim == 3:
            normalized.append(embed)
            continue
        if embed.ndim != 2:
            raise ValueError(f"Expected 2D or 3D image embeds, got {tuple(embed.shape)}")
        batch_size = int(mask.shape[0])
        hidden = embed.shape[-1]
        if embed.shape[0] % batch_size != 0:
            raise ValueError(
                f"Cannot reshape flattened image embeds {tuple(embed.shape)} "
                f"for batch_size={batch_size}"
            )
        num_tokens = embed.shape[0] // batch_size
        normalized.append(embed.reshape(batch_size, num_tokens, hidden))
    return normalized


def _pack_concat_segments(
    segments: list[tuple[torch.Tensor, torch.Tensor, int]],
    *,
    make_att_2d_masks,
    prepare_attention_mask_4d=None,
) -> dict[str, torch.Tensor]:
    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []
    token_types: list[int] = []

    for segment_embs, segment_mask, att_type in segments:
        segment_mask = segment_mask.to(device=segment_embs.device, dtype=torch.bool)
        embs.append(segment_embs)
        pad_masks.append(segment_mask)
        token_types += [int(att_type)] * segment_embs.shape[1]

    inputs_embeds = torch.cat(embs, dim=1)
    pad_mask = torch.cat(pad_masks, dim=1)
    token_type_ids = torch.tensor(
        token_types,
        dtype=torch.int64,
        device=inputs_embeds.device,
    )[None, :].expand(inputs_embeds.shape[0], -1)

    att_2d = make_att_2d_masks(pad_mask, token_type_ids.to(dtype=torch.bool))
    if prepare_attention_mask_4d is None:
        attention_mask = att_2d
    else:
        attention_mask = prepare_attention_mask_4d(att_2d)

    position_ids = torch.cumsum(pad_mask, dim=1) - 1

    return {
        "inputs_embeds": inputs_embeds,
        "pad_mask": pad_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
    }


@torch.no_grad()
def compact_pi05_prefix(packed: dict[str, Any]) -> dict[str, Any]:
    """Strip padding columns from a PI0.5 prefix (fixed static length for Edge export)."""
    pad_mask = packed["pad_mask"]
    position_ids = packed["position_ids"]
    inputs_embeds = packed["inputs_embeds"]

    valid = pad_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "compact_pi05_prefix requires equal valid token counts across the batch"
        )

    compact_len = int(valid_counts[0].item())
    compact_embs = torch.stack(
        [inputs_embeds[b, valid[b], :] for b in range(inputs_embeds.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [position_ids[b, valid[b]] for b in range(position_ids.shape[0])],
        dim=0,
    )

    result: dict[str, Any] = {
        "inputs_embeds": compact_embs,
        "pad_mask": torch.ones(
            inputs_embeds.shape[0],
            compact_len,
            device=pad_mask.device,
            dtype=torch.bool,
        ),
        "attention_mask": torch.zeros(
            inputs_embeds.shape[0],
            1,
            compact_len,
            compact_len,
            device=inputs_embeds.device,
            dtype=torch.float32,
        ),
        "position_ids": compact_position_ids,
    }

    if packed.get("image_token_mask") is not None:
        image_token_mask = packed["image_token_mask"]
        result["image_token_mask"] = torch.stack(
            [image_token_mask[b, valid[b]] for b in range(image_token_mask.shape[0])],
            dim=0,
        )

    if packed.get("token_type_ids") is not None:
        token_type_ids = packed["token_type_ids"]
        result["token_type_ids"] = torch.stack(
            [token_type_ids[b, valid[b]] for b in range(token_type_ids.shape[0])],
            dim=0,
        )

    return result


@torch.no_grad()
def pack_pi05_prefix(
    core,
    image_embs: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    *,
    inputs_dtype: torch.dtype | None = None,
    compact: bool = True,
) -> dict[str, Any]:
    """PI0.5: concat per-camera vision tokens then text → [B, L, H]."""
    image_embs = _normalize_pi05_image_embs(image_embs, img_masks)

    segments: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for image_emb, image_mask in zip(image_embs, img_masks, strict=True):
        bsz, num_image_tokens = image_emb.shape[:2]
        expanded_mask = image_mask[:, None].expand(bsz, num_image_tokens)
        segments.append((image_emb, expanded_mask, 0))

    text_embs = core.paligemma_with_expert.embed_language_tokens(tokens)
    segments.append((text_embs, masks.to(dtype=torch.bool), 0))

    packed = _pack_concat_segments(
        segments,
        make_att_2d_masks=pi05_make_att_2d_masks,
        prepare_attention_mask_4d=core._prepare_attention_masks_4d,
    )

    if inputs_dtype is not None:
        packed["inputs_embeds"] = packed["inputs_embeds"].to(inputs_dtype)

    if compact:
        return compact_pi05_prefix(packed)
    return packed


@torch.no_grad()
def pack_groot_language_inputs(
    core,
    vit_embs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """GR00T: embed input_ids, scatter vision rows into image-token placeholder slots."""
    eagle = core.backbone.eagle_model
    image_token_id = int(
        getattr(eagle, "image_token_index", eagle.config.image_token_index)
    )
    token_embed_fn = eagle.language_model.get_input_embeddings()

    input_embs = token_embed_fn(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    attention_mask = attention_mask.to(device=input_embs.device)

    bsz, seq_len, hidden = input_embs.shape
    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)

    image_token_mask = flat_ids == image_token_id
    flat_image_embs = vit_embs.reshape(-1, hidden).to(
        device=flat_embs.device,
        dtype=flat_embs.dtype,
    )

    num_slots = int(image_token_mask.sum().item())
    if flat_image_embs.shape[0] < num_slots:
        raise ValueError(
            f"Not enough image embeddings for placeholders: "
            f"{flat_image_embs.shape[0]} embeddings for {num_slots} slots"
        )

    flat_embs[image_token_mask] = flat_image_embs[:num_slots]
    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden)
    pad_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

    return {
        "inputs_embeds": inputs_embeds,
        "pad_mask": pad_mask,
        "attention_mask": attention_mask,
        "position_ids": torch.cumsum(pad_mask, dim=1) - 1,
        "image_token_mask": image_token_mask.reshape(bsz, seq_len),
    }


@torch.no_grad()
def pack_smolvla_prefix(
    core,
    image_embs: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    state: torch.Tensor,
) -> dict[str, Any]:
    """SmolVLA: per-camera vision (+ optional special tokens), text, and state → [B, L, H]."""
    if len(image_embs) != len(img_masks):
        raise ValueError(
            f"image_embs and img_masks length mismatch: {len(image_embs)} vs {len(img_masks)}"
        )

    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []
    att_masks: list[int] = []

    for img_emb, img_mask in zip(image_embs, img_masks, strict=True):
        batch_size = int(img_emb.shape[0])
        if core.add_image_special_tokens:
            image_start_token = (
                core.vlm_with_expert.embed_language_tokens(
                    core.global_image_start_token.to(device=tokens.device)
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            image_start_mask = torch.ones_like(
                image_start_token[:, :, 0],
                dtype=torch.bool,
                device=image_start_token.device,
            )
            embs.append(image_start_token)
            pad_masks.append(image_start_mask)
            att_masks += [0] * image_start_mask.shape[1]

        img_emb = img_emb * torch.tensor(
            img_emb.shape[-1] ** 0.5,
            dtype=img_emb.dtype,
            device=img_emb.device,
        )

        num_img_embs = int(img_emb.shape[1])
        expanded_img_mask = img_mask[:, None].expand(batch_size, num_img_embs)
        embs.append(img_emb)
        pad_masks.append(expanded_img_mask)
        att_masks += [0] * num_img_embs

        if core.add_image_special_tokens:
            image_end_token = (
                core.vlm_with_expert.embed_language_tokens(
                    core.image_end_token.to(device=tokens.device)
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            image_end_mask = torch.ones_like(
                image_end_token[:, :, 0],
                dtype=torch.bool,
                device=image_end_token.device,
            )
            embs.append(image_end_token)
            pad_masks.append(image_end_mask)
            att_masks += [0] * image_end_mask.shape[1]

    lang_emb = core.vlm_with_expert.embed_language_tokens(tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    embs.append(lang_emb)
    pad_masks.append(masks)
    att_masks += [0] * lang_emb.shape[1]

    state_emb = core.state_proj(state)
    state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
    embs.append(state_emb)
    pad_masks.append(torch.ones(state_emb.shape[:2], dtype=torch.bool, device=state_emb.device))
    att_masks += [1] * state_emb.shape[1]

    inputs_embeds = torch.cat(embs, dim=1)
    pad_mask = torch.cat(pad_masks, dim=1)
    prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_mask.device)[
        None, :
    ].expand(inputs_embeds.shape[0], -1)

    if core.prefix_length > 0 and pad_mask.shape[1] < core.prefix_length:
        inputs_embeds = pad_tensor(inputs_embeds, core.prefix_length, pad_value=0)
        pad_mask = pad_tensor(pad_mask, core.prefix_length, pad_value=0)
        prefix_att_masks = pad_tensor(prefix_att_masks, core.prefix_length, pad_value=0)

    return {
        "inputs_embeds": inputs_embeds,
        "pad_mask": pad_mask,
        "attention_mask": smolvla_make_att_2d_masks(pad_mask, prefix_att_masks),
        "position_ids": torch.cumsum(pad_mask, dim=1) - 1,
    }

