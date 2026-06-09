from dataclasses import dataclass, replace
from typing import Callable, Literal

import torch

PackingStyle = Literal["concat_prefix", "placeholder_replace"]

@dataclass
class PackedLanguageInputs:
    inputs_embeds: torch.Tensor
    pad_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    image_token_mask: torch.Tensor | None = None

    def as_tuple(self):
        return (
            self.inputs_embeds,
            self.pad_mask,
            self.attention_mask,
            self.position_ids,
        )

    def with_inputs_embeds(self, inputs_embeds: torch.Tensor) -> "PackedLanguageInputs":
        return replace(self, inputs_embeds=inputs_embeds)

    def to(
        self,
        *,
        device: torch.device | None = None,
        inputs_dtype: torch.dtype | None = None,
    ) -> "PackedLanguageInputs":
        def move(tensor, *, dtype=None):
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            return tensor.to(**kwargs) if kwargs else tensor

        return PackedLanguageInputs(
            inputs_embeds=move(self.inputs_embeds, dtype=inputs_dtype),
            pad_mask=None if self.pad_mask is None else move(self.pad_mask),
            attention_mask=None if self.attention_mask is None else move(self.attention_mask),
            position_ids=None if self.position_ids is None else move(self.position_ids),
            image_token_mask=None if self.image_token_mask is None else move(self.image_token_mask),
        )

@torch.no_grad()
def embed_images(images, *, eager_embed_fn=None, vision_runner=None):
    if vision_runner is None:
        if eager_embed_fn is None:
            raise ValueError("eager_embed_fn is required when vision_runner is None")
        return [eager_embed_fn(image) for image in images]

    return [vision_runner(image) for image in images]

@torch.no_grad()
def pack_concat_prefix(
    *,
    image_embs: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    text_embs: torch.Tensor,
    text_mask: torch.Tensor,
    make_att_2d_masks: Callable,
    prepare_attention_mask_4d: Callable,
) -> PackedLanguageInputs:
    embs = []
    pad_masks = []
    att_masks = []

    for img_emb, img_mask in zip(image_embs, image_masks, strict=True):
        bsz, n_img = img_emb.shape[:2]

        embs.append(img_emb)
        pad_masks.append(
            img_mask.to(device=img_emb.device, dtype=torch.bool)[:, None].expand(bsz, n_img)
        )
        att_masks += [0] * n_img

    embs.append(text_embs)
    pad_masks.append(text_mask.to(device=text_embs.device, dtype=torch.bool))
    att_masks += [0] * text_embs.shape[1]

    inputs_embeds = torch.cat(embs, dim=1)
    pad_mask = torch.cat(pad_masks, dim=1)

    att_mask_1d = torch.tensor(
        att_masks,
        dtype=torch.bool,
        device=inputs_embeds.device,
    )[None, :].expand(inputs_embeds.shape[0], -1)

    att_mask_2d = make_att_2d_masks(pad_mask, att_mask_1d)
    attention_mask = prepare_attention_mask_4d(att_mask_2d)
    position_ids = torch.cumsum(pad_mask, dim=1) - 1

    return PackedLanguageInputs(
        inputs_embeds=inputs_embeds,
        pad_mask=pad_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

@torch.no_grad()
def pack_placeholder_replace(
    *,
    input_ids: torch.Tensor,
    token_embed_fn: Callable,
    image_embs: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor,
) -> PackedLanguageInputs:
    input_embs = token_embed_fn(input_ids)
    attention_mask = attention_mask.to(device=input_embs.device)

    bsz, seq_len, hidden = input_embs.shape
    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)

    image_token_mask = flat_ids == image_token_id
    flat_image_embs = image_embs.reshape(-1, hidden).to(
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
    position_ids = torch.cumsum(pad_mask, dim=1) - 1

    return PackedLanguageInputs(
        inputs_embeds=inputs_embeds,
        pad_mask=pad_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        image_token_mask=image_token_mask.reshape(bsz, seq_len),
    )

@torch.no_grad()
def compact_packed_language_inputs(packed: PackedLanguageInputs) -> PackedLanguageInputs:
    if packed.pad_mask is None:
        raise ValueError("packed.pad_mask is required for compaction")
    if packed.position_ids is None:
        raise ValueError("packed.position_ids is required for compaction")

    valid = packed.pad_mask.to(device=packed.inputs_embeds.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)

    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "compact_packed_language_inputs requires equal valid token counts across the batch"
        )

    compact_len = int(valid_counts[0].item())
    compact_embs = torch.stack(
        [packed.inputs_embeds[b, valid[b], :] for b in range(packed.inputs_embeds.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [packed.position_ids[b, valid[b]] for b in range(packed.position_ids.shape[0])],
        dim=0,
    )

    compact_image_token_mask = None
    if packed.image_token_mask is not None:
        compact_image_token_mask = torch.stack(
            [packed.image_token_mask[b, valid[b]] for b in range(packed.image_token_mask.shape[0])],
            dim=0,
        )

    compact_pad_mask = torch.ones(
        packed.inputs_embeds.shape[0],
        compact_len,
        device=packed.pad_mask.device,
        dtype=torch.bool,
    )
    compact_attention_mask = torch.zeros(
        packed.inputs_embeds.shape[0],
        1,
        compact_len,
        compact_len,
        device=packed.inputs_embeds.device,
        dtype=torch.float32,
    )

    return PackedLanguageInputs(
        inputs_embeds=compact_embs,
        pad_mask=compact_pad_mask,
        attention_mask=compact_attention_mask,
        position_ids=compact_position_ids,
        image_token_mask=compact_image_token_mask,
    )
