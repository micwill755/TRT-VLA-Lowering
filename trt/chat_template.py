"""VitRunner ``processed_chat_template.json`` builders for Edge-LLM export.

``llm_inference`` reads this file at runtime so ``textPreprocess`` can expand each
image placeholder in the chat prompt to ``seq_len`` vision-token slots (matching
``visual.engine`` output). Pair with ``save_tokenizer_for_edge_llm`` in ``language/``.
"""

from __future__ import annotations

from typing import Any


# One placeholder per camera; VitRunner::textPreprocess expands each to builder_config.seq_len.
def build_groot_vitrunner_chat_template(tokenizer) -> dict[str, Any]:
    """Build processed_chat_template.json for VitRunner (single image placeholder per image)."""
    im_start = "<|im_start|>"
    im_end = tokenizer.eos_token
    if not im_end:
        raise ValueError("Tokenizer eos_token is required to build the GROOT chat template.")

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

    system_prefix = system_only.split("SYS", 1)[0]
    system_suffix = "SYS" + system_only.split("SYS", 1)[1]

    user_prefix = user_only.split("TEXTONLY", 1)[0]
    user_suffix = "TEXTONLY" + user_only.split("TEXTONLY", 1)[1]

    assistant_only = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "TEXTONLY"},
            {"role": "assistant", "content": "ASSIST"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
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
        "content_types": {
            "image": {"format": "<img><IMG_CONTEXT></img>"},
        },
        "generation_prompt": generation_prompt,
        "default_system_prompt": "You are a helpful assistant.",
    }
    
def build_pi05_vitrunner_chat_template(*, image_format: str = "<image>") -> dict[str, Any]:
    """Minimal processed_chat_template.json for VitRunner image placeholder expansion."""
    return {
        "model_path": "pi05-vitrunner",
        "roles": {
            "user": {"prefix": "", "suffix": ""},
        },
        "content_types": {
            "image": {"format": image_format},
        },
        "generation_prompt": "",
        "default_system_prompt": "",
    }

def build_smolvla_vitrunner_chat_template(tokenizer, *, image_token_id: int) -> dict[str, Any]:
    """Minimal processed_chat_template.json for VitRunner image placeholder expansion."""
    image_format = tokenizer.decode([int(image_token_id)])
    if not image_format.strip():
        image_format = "<image>"
    return {
        "model_path": "smolvla-vitrunner",
        "roles": {
            "user": {"prefix": "", "suffix": ""},
        },
        "content_types": {
            "image": {"format": image_format},
        },
        "generation_prompt": "",
        "default_system_prompt": "",
    }
