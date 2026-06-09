from dataclasses import dataclass, replace
from typing import Callable, Literal

import torch

PackingStyle = Literal[
    "concat_prefix",
    "placeholder_replace",
    "chat_template_placeholder",
    "concat_typed_regions",
]

@dataclass
class PromptTensorInputs:
    text_embs: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    image_embs: list[torch.Tensor] | torch.Tensor | None = None
    image_masks: list[torch.Tensor] | None = None
    state_embs: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

@dataclass
class PromptPackingSpec:
    style: PackingStyle
    image_token_id: int | None = None
    token_embed_fn: Callable | None = None
    make_att_2d_masks: Callable | None = None
    prepare_attention_mask_4d: Callable | None = None
    image_att_type: int = 0
    text_att_type: int = 0
    state_att_type: int = 1

@dataclass
class PackedLanguageInputs:
    inputs_embeds: torch.Tensor
    pad_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    image_token_mask: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None

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
            token_type_ids=None if self.token_type_ids is None else move(self.token_type_ids),
        )

def _require(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required")
    return value

def _as_image_list(image_embs: list[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(image_embs, torch.Tensor):
        return [image_embs]
    return list(image_embs)

def _cat_image_embs(image_embs: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(image_embs, torch.Tensor):
        return image_embs
    if len(image_embs) == 1:
        return image_embs[0]
    return torch.cat(image_embs, dim=1)

class MultimodalPromptProcessor:
    def __init__(self, spec: PromptPackingSpec):
        self.spec = spec

    @torch.no_grad()
    def __call__(self, inputs: PromptTensorInputs) -> PackedLanguageInputs:
        if self.spec.style == "concat_prefix":
            return self._pack_concat_prefix(inputs)
        if self.spec.style == "placeholder_replace":
            return self._pack_placeholder_replace(inputs)
        if self.spec.style == "chat_template_placeholder":
            return self._pack_chat_template_placeholder(inputs)
        if self.spec.style == "concat_typed_regions":
            return self._pack_concat_typed_regions(inputs)
        raise NotImplementedError(f"Unsupported packing style: {self.spec.style}")

    def _build_concat_output(
        self,
        segments: list[tuple[torch.Tensor, torch.Tensor, int]],
    ) -> PackedLanguageInputs:
        make_att_2d_masks = _require(self.spec.make_att_2d_masks, "make_att_2d_masks")

        embs = []
        pad_masks = []
        token_types = []

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
        if self.spec.prepare_attention_mask_4d is None:
            attention_mask = att_2d
        else:
            attention_mask = self.spec.prepare_attention_mask_4d(att_2d)

        position_ids = torch.cumsum(pad_mask, dim=1) - 1

        return PackedLanguageInputs(
            inputs_embeds=inputs_embeds,
            pad_mask=pad_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

    def _image_segments(self, inputs: PromptTensorInputs) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        image_embs = _as_image_list(_require(inputs.image_embs, "image_embs"))
        image_masks = _require(inputs.image_masks, "image_masks")

        segments = []
        for image_emb, image_mask in zip(image_embs, image_masks, strict=True):
            bsz, num_image_tokens = image_emb.shape[:2]
            expanded_mask = image_mask[:, None].expand(bsz, num_image_tokens)
            segments.append((image_emb, expanded_mask, self.spec.image_att_type))
        return segments

    def _pack_concat_prefix(self, inputs: PromptTensorInputs) -> PackedLanguageInputs:
        text_embs = _require(inputs.text_embs, "text_embs")
        text_mask = _require(inputs.text_mask, "text_mask")

        segments = self._image_segments(inputs)
        segments.append((text_embs, text_mask, self.spec.text_att_type))
        return self._build_concat_output(segments)

    def _pack_concat_typed_regions(self, inputs: PromptTensorInputs) -> PackedLanguageInputs:
        text_embs = _require(inputs.text_embs, "text_embs")
        text_mask = _require(inputs.text_mask, "text_mask")

        segments = self._image_segments(inputs)
        segments.append((text_embs, text_mask, self.spec.text_att_type))

        if inputs.state_embs is not None:
            state_embs = inputs.state_embs
            if state_embs.ndim == 2:
                state_embs = state_embs[:, None, :]

            if inputs.state_mask is None:
                state_mask = torch.ones(
                    state_embs.shape[:2],
                    dtype=torch.bool,
                    device=state_embs.device,
                )
            else:
                state_mask = inputs.state_mask

            segments.append((state_embs, state_mask, self.spec.state_att_type))

        return self._build_concat_output(segments)

    def _pack_chat_template_placeholder(self, inputs: PromptTensorInputs) -> PackedLanguageInputs:
        # The chat/template stage has already produced input_ids containing image placeholder slots.
        return self._pack_placeholder_replace(inputs)

    def _pack_placeholder_replace(self, inputs: PromptTensorInputs) -> PackedLanguageInputs:
        input_ids = _require(inputs.input_ids, "input_ids")
        image_embs = _cat_image_embs(_require(inputs.image_embs, "image_embs"))
        token_embed_fn = _require(self.spec.token_embed_fn, "token_embed_fn")
        image_token_id = _require(self.spec.image_token_id, "image_token_id")

        input_embs = token_embed_fn(input_ids)
        attention_mask = inputs.attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
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

    compact_token_type_ids = None
    if packed.token_type_ids is not None:
        compact_token_type_ids = torch.stack(
            [packed.token_type_ids[b, valid[b]] for b in range(packed.token_type_ids.shape[0])],
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
        token_type_ids=compact_token_type_ids,
    )
