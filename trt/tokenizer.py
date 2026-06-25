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

from __future__ import annotation

from typing import Any

import json
import pathlib
import torch.nn as nn

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
