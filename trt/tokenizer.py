"""Tokenizer and embedding sidecars for TensorRT-Edge-LLM ``llm_inference``.

``language.engine`` only runs the causal LM on ``inputs_embeds``. The C++
runtime loads the rest from ``language/`` before that forward:

- HF tokenizer assets — decode prompts and build ``input_ids``
- ``processed_chat_template.json`` — VitRunner ``textPreprocess`` expands image
  placeholders to ``builder_config.seq_len`` vision slots per image
- ``embedding.safetensors`` — ``embeddingLookupWithImageInsertion`` maps token
  IDs to vectors and scatters ``visual_embeds`` from ``visual.engine``

Export must always write these files; ``llm_inference`` will not start without them.
"""

from __future__ import annotations

from typing import Any

import json
import pathlib
import torch.nn as nn

def format_edge_llm_prompt(
    messages: list[dict[str, Any]],
    chat_template: dict[str, Any],
    *,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages the way ``Tokenizer::applyChatTemplate`` does in Edge-LLM."""
    roles = chat_template["roles"]
    content_types = chat_template.get("content_types", {})
    formatted_complete = ""

    system_prompt = ""
    has_explicit_system = False
    if messages and messages[0].get("role") == "system":
        has_explicit_system = True
        content = messages[0]["content"]
        if isinstance(content, str):
            system_prompt = content
        else:
            for item in content:
                if item.get("type") == "text":
                    system_prompt += item.get("text", item.get("content", ""))
    elif apply_chat_template and chat_template.get("default_system_prompt"):
        has_explicit_system = True
        system_prompt = str(chat_template["default_system_prompt"])

    if system_prompt or (has_explicit_system and apply_chat_template):
        if apply_chat_template:
            sys_role = roles["system"]
            formatted_complete = (
                sys_role["prefix"] + system_prompt + sys_role["suffix"]
            )
        else:
            formatted_complete = system_prompt

    start_idx = 1 if messages and messages[0].get("role") == "system" else 0
    for idx in range(start_idx, len(messages)):
        message = messages[idx]
        role = message["role"]
        role_fmt = roles[role]
        formatted_message = role_fmt["prefix"] if apply_chat_template else ""

        content = message["content"]
        items = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        for item in items:
            content_type = item.get("type", "text")
            if content_type == "text":
                formatted_message += item.get("text", item.get("content", ""))
            elif content_type in content_types:
                formatted_message += content_types[content_type]["format"]

        if apply_chat_template:
            formatted_message += role_fmt["suffix"]
        formatted_complete += formatted_message

    if add_generation_prompt:
        formatted_complete += chat_template.get("generation_prompt", "")

    return formatted_complete


def vitrunner_expand_image_tokens(
    input_ids: list[int],
    *,
    image_token_id: int,
    seq_len_per_image: int,
    vocab_size: int,
) -> list[int]:
    """Expand compact image placeholders like ``VitRunner::textPreprocess``."""
    next_image_token_id = vocab_size
    expanded: list[int] = []
    for token_id in input_ids:
        if token_id == image_token_id:
            for _ in range(seq_len_per_image):
                expanded.append(next_image_token_id)
                next_image_token_id += 1
        else:
            expanded.append(token_id)
    return expanded


def edge_llm_tokenize_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    chat_template: dict[str, Any],
    *,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Tokenize + expand image slots for Edge-LLM ``llm_inference`` / VitRunner."""
    prompt = format_edge_llm_prompt(
        messages,
        chat_template,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    compact_ids = tokenizer.encode(prompt)
    return vitrunner_expand_image_tokens(
        compact_ids,
        image_token_id=int(chat_template["image_token_id"]),
        seq_len_per_image=int(chat_template["seq_len_per_image"]),
        vocab_size=len(tokenizer),
    )


def groot_edge_chat_template(
    *,
    image_token_id: int,
    seq_len_per_image: int,
    im_end: str = "",
) -> dict[str, Any]:
    """Chat template JSON consumed by ``llm_inference`` for GR00T/Eagle prompts."""
    role_suffix = f"{im_end}\n"
    return {
        "roles": {
            "system": {
                "prefix": "<|im_start|>system\n",
                "suffix": role_suffix,
            },
            "user": {
                "prefix": "<|im_start|>user\n",
                "suffix": role_suffix,
            },
            "assistant": {
                "prefix": "<|im_start|>assistant\n",
                "suffix": role_suffix,
            },
        },
        "content_types": {
            "image": {"format": "<img><IMG_CONTEXT></img>"},
        },
        "generation_prompt": "<|im_start|>assistant\n",
        "default_system_prompt": "You are a helpful assistant",
        "image_token_id": int(image_token_id),
        "seq_len_per_image": int(seq_len_per_image),
    }


def pi05_edge_chat_template(*, max_seq_len: int) -> dict[str, Any]:
    """Sidecar template for PI0.5 compact-prefix language engines."""
    return {
        "model_path": "pi05",
        "roles": {
            "system": {"prefix": "", "suffix": ""},
            "user": {"prefix": "", "suffix": "\n"},
            "assistant": {"prefix": "", "suffix": ""},
        },
        "content_types": {
            "image": {"format": "<image>"},
        },
        "generation_prompt": "",
        "default_system_prompt": "",
        "prefix_strategy": "pi05_compact_prefix",
        "max_seq_len": int(max_seq_len),
    }


def save_tokenizer_for_edge_llm(
    language_engine_dir: str | pathlib.Path,
    *,
    tokenizer: Any,
    chat_template: dict[str, Any],
) -> None:
    """Write HF tokenizer assets and ``processed_chat_template.json`` into ``language/``.

    Required by ``llm_inference`` for prompt tokenization and VitRunner image-slot
    expansion. The tokenizer is always loaded in export ``main`` and passed through;
    there is no optional fallback path.
    """
    dst = pathlib.Path(language_engine_dir)
    dst.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(dst)
    (dst / "processed_chat_template.json").write_text(
        json.dumps(chat_template, indent=2) + "\n"
    )

def save_embedding_table(language_model: nn.Module, output_dir: str | pathlib.Path) -> None:
    """Write ``embedding.safetensors`` for C++ token embedding lookup.

    ``llm_inference`` uses this table (not the TRT graph) to build ``inputs_embeds``
    and insert vision rows at image-token positions before ``language.engine`` runs.
    """
    from safetensors.torch import save_file

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_tokens = language_model.get_input_embeddings()
    embedding_weight = embed_tokens.weight.data.detach().cpu().half()
    save_file({"embedding": embedding_weight}, output_dir / "embedding.safetensors")
