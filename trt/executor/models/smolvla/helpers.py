from __future__ import annotations

import math

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def build_smolvla_prefix_embs(
    smolvla_model,
    img_masks,
    lang_tokens,
    lang_masks,
    image_embs,
    images,
    state,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact image+language+state prefix embeddings for SmolVLA language prefill."""
    per_camera_batch = int(images[0].shape[0])
    image_embs_list = list(
        image_embs.reshape(len(images), per_camera_batch, -1, image_embs.shape[-1])
    )

    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []

    for img_emb, img_mask in zip(image_embs_list, img_masks, strict=True):
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * (img_emb_dim**0.5)
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        img_mask = img_mask.to(device=img_emb.device, dtype=torch.bool)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

    lang_emb = smolvla_model.vlm_with_expert.embed_language_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)
    embs.append(lang_emb)
    pad_masks.append(lang_masks.to(device=lang_emb.device, dtype=torch.bool))

    state_emb = smolvla_model.state_proj(state)
    state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
    embs.append(state_emb)
    bsize = state_emb.shape[0]
    device = state_emb.device
    states_seq_len = state_emb.shape[1]
    pad_masks.append(torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device))

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    valid = prefix_pad_masks.to(device=prefix_embs.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "build_smolvla_prefix_embs requires equal valid token counts across the batch"
        )

    compact_len = int(valid_counts[0].item())
    compact_embs = torch.stack(
        [prefix_embs[b, valid[b], :] for b in range(prefix_embs.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [prefix_position_ids[b, valid[b]] for b in range(prefix_position_ids.shape[0])],
        dim=0,
    )
    compact_pad_mask = torch.ones(
        prefix_embs.shape[0],
        compact_len,
        device=prefix_pad_masks.device,
        dtype=torch.bool,
    )
    compact_attention_mask = torch.zeros(
        prefix_embs.shape[0],
        1,
        compact_len,
        compact_len,
        device=prefix_embs.device,
        dtype=compact_embs.dtype,
    )
    return compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids


def make_smolvla_suffix_position_and_mask(model, prefix_pad_masks, x_t, timestep):
    """Suffix position ids and full 2D attention mask for one denoise step."""
    _, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_masks = prefix_pad_masks.to(device=x_t.device)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, full_att_2d_masks
