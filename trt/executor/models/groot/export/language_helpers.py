from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from trt.profile import ClonedLanguageSubgraph
from trt.utils import clone_hf_module_for_export


def clone_language_subgraph(
    model: nn.Module,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float16,
) -> ClonedLanguageSubgraph:
    source_lm = model.backbone.eagle_model.language_model
    source_decoder = getattr(source_lm, "model", source_lm)

    lm_config = copy.deepcopy(source_lm.config)
    lm_config.num_hidden_layers = len(source_decoder.layers)

    language_model = clone_hf_module_for_export(
        source_lm,
        device,
        dtype=dtype,
        config=lm_config,
    )
    decoder = getattr(language_model, "model", language_model)
    lm_head = clone_hf_module_for_export(language_model.lm_head, device, dtype=dtype)
    return ClonedLanguageSubgraph(
        language_model=language_model,
        decoder=decoder,
        lm_head=lm_head,
        config=language_model.config,
    )


@torch.no_grad()
def pack_language_export_inputs(
    model: nn.Module,
    *,
    image_embs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    eagle = model.backbone.eagle_model
    image_token_id = int(getattr(eagle, "image_token_index", eagle.config.image_token_index))
    token_embed_fn = eagle.language_model.get_input_embeddings()

    input_embs = token_embed_fn(input_ids)
    attention_mask = attention_mask.to(device=input_embs.device, dtype=torch.long)

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
            f"Not enough image embeddings: {flat_image_embs.shape[0]} for {num_slots} slots"
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


def build_language_chat_template(tokenizer: Any) -> dict[str, Any]:
    im_end = tokenizer.eos_token
    if not im_end:
        raise ValueError("Tokenizer eos_token required")

    system_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": "SYS"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TEXTONLY"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    with_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TEXTONLY"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assistant_only = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "TEXTONLY"},
            {"role": "assistant", "content": "ASSIST"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    system_prefix = system_only.split("SYS", 1)[0]
    system_suffix = "SYS" + system_only.split("SYS", 1)[1]
    user_prefix = user_only.split("TEXTONLY", 1)[0]
    user_suffix = "TEXTONLY" + user_only.split("TEXTONLY", 1)[1]
    assistant_prefix = assistant_only[len(user_only) :].split("ASSIST", 1)[0]
    assistant_suffix = "ASSIST" + assistant_only.split("ASSIST", 1)[1]
    generation_prompt = with_gen[len(user_only) :]

    return {
        "model_path": "groot-vitrunner",
        "roles": {
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "assistant": {"prefix": assistant_prefix, "suffix": assistant_suffix},
        },
        "content_types": {"image": {"format": "<img><IMG_CONTEXT></img>"}},
        "generation_prompt": generation_prompt,
        "default_system_prompt": "You are a helpful assistant.",
    }
