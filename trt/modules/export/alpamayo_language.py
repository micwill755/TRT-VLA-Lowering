from __future__ import annotations

import torch
import torch.nn as nn


def build_alpamayo_prefix_embs(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    image_embs: torch.Tensor,
    *,
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Alpamayo LM prefix embeds from text tokens + packed TRT vision features.

    Mirrors PI05's ``build_pi05_prefix_embs``: language embeds come from the
    embedding table, vision features are raw tensors, and fusion is plain torch
    ops (no HF ``get_placeholder_mask`` / ``get_image_features``).

    Alpamayo differs from PI05 in layout — vision slots are interleaved image
    placeholder tokens inside ``input_ids``, so we scatter rather than concat.

    Returns
    -------
    inputs_embeds
        ``[B, S, H]`` text embeds with vision features scattered in.
    visual_pos_masks
        ``[B, S]`` bool mask of vision placeholder positions.
    """
    inputs_embeds = embed_tokens(input_ids)
    image_cat = image_embs.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    if image_cat.dim() > 2:
        image_cat = image_cat.reshape(-1, image_cat.shape[-1])

    image_mask = (input_ids == int(image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
    n_slots = int(image_mask[..., 0].sum().item())
    if n_slots != int(image_cat.shape[0]):
        raise ValueError(
            f"image placeholder slots ({n_slots}) != packed vision tokens "
            f"({int(image_cat.shape[0])})"
        )

    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_cat)
    return inputs_embeds, image_mask[..., 0]
