"""GR00T language export hooks: Eagle LM prefill → TRT LLMEngineRunner engine.

Stage 1 in the GROOT export pipeline (after vision). ``glue.vision_to_language`` wires
stage-0 metadata into ``stage_inputs``; ``plan_export`` builds an ``ExportPlan``;
``compile`` traces ``CausalLMExportModule`` and writes ``language/language.engine``;
``save_artifacts`` writes embedding table + tokenizer JSON for C++ runtime.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.compile import save_trt_engine_module
from trt.hooks.export.plan import ExportPlan
from trt.language import (
    compute_vit_expanded_seq_len,
    language_edge_output_names,
    language_edge_trt_settings,
    language_head_dim,
    make_language_edge_flat_tensors,
    make_language_edge_input_specs,
)
from trt.modules.export.language import CausalLMExportModule
from trt.plugin.plugin_utils import patch_language_attention, restore_attention
from trt.rope import make_rope_rotary_cos_sin
from trt.context import EdgeContext
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm
from trt.utils import clone_hf_module_for_export

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    eagle = ctx.model.backbone.eagle_model
    language = eagle.language_model
    device, dtype = ctx.device, ctx.dtype

    # --- upstream vision output: [B * S_visual, H_lm] ---
    image_embs = inputs["tensors"]["image_embs"]

    # --- tokenized prompt (pipeline-level preprocess) ---
    input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = inputs.get("attention_mask")

    image_token_index = getattr(
        eagle, "image_token_index", eagle.config.image_token_index
    )

    # --- build packed language embeddings (splice vision rows into token embeds) ---
    input_embs = language.get_input_embeddings()(input_ids)
    bsz, seq_len, hidden = input_embs.shape

    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)
    image_token_mask = flat_ids == image_token_index

    flat_image_embs = image_embs.reshape(-1, hidden).to(
        device=flat_embs.device,
        dtype=flat_embs.dtype,   # match embedding dtype, not hardcoded fp16
    )
    num_slots = int(image_token_mask.sum().item())
    flat_embs[image_token_mask] = flat_image_embs[:num_slots]

    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden).to(device=device).contiguous()
    lm_dtype = next(language.parameters()).dtype

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # --- EAGER: stock HF path (match model weight dtype, test_vla rung A) ---
    eager_inputs = {
        "inputs_embeds": inputs_embeds.to(dtype=lm_dtype).contiguous(),
        "attention_mask": attention_mask.to(device=device),
        "position_ids": position_ids,
    }

    # --- TRT/SERIALIZED: flat Edge-LLM bindings (ctx.dtype for engine ABI) ---
    trt_inputs_embeds = inputs_embeds.to(dtype=dtype).contiguous()
    # --- config dims ---
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))

    decoder = getattr(language, "model", language)   # Qwen3Model with .layers
    num_layers = len(decoder.layers)

    # --- trace target: manual decoder loop + lm_head (PluginAttention inside) ---
    lm_module = CausalLMExportModule(
        decoder,
        language.lm_head,
        select_layer=-1,
    ).eval().to(device=device)

    # --- flat Edge-LLM bindings ---
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        device,
        language_model=language,
        position_ids=position_ids,
    )

    ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.int64)

    kv_caches = [
        torch.zeros(
            bsz,
            2,
            num_key_value_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)  # fresh prefill

    lm_inputs = (
        trt_inputs_embeds,      # spliced vision rows — NOT raw input_embs
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    return {
        "language_model": language,
        "eager_inputs": eager_inputs,
        "lm_inputs": lm_inputs,
        "lm_module": lm_module,
        "decoder": decoder,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
    }

def export(ctx: EdgeContext, inputs: dict) -> dict:
    language = inputs["language_model"]
    lm_module = inputs["lm_module"]
    lm_inputs = inputs["lm_inputs"]
    decoder = inputs["decoder"]

    batch_size = int(inputs["batch_size"])
    seq_len = int(inputs["seq_len"])
    hidden_size = int(inputs["hidden_size"])
    num_hidden_layers = int(inputs["num_hidden_layers"])
    num_attention_heads = int(inputs["num_attention_heads"])
    num_key_value_heads = int(inputs["num_key_value_heads"])
    head_dim = int(inputs["head_dim"])

    input_names = [
        "inputs_embeds",
        "rope_rotary_cos_sin",
        "context_lengths",
        "kvcache_start_index",
        "last_token_ids",
    ] + [f"past_key_values_{i}" for i in range(num_hidden_layers)]
    output_names = ["logits", "lm_hidden_states", "prefix_k", "prefix_v"]

    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        enable_bidirectional_prefill=0,
    )

    try:
        engine_path = save_trt_engine_module(
            lm_module,
            lm_inputs,
            ctx.engine_root / "language",
            engine_file="language.engine",
            model_type="language",
            component="language",
            input_names=input_names,
            output_names=output_names,
            example_output=None,
            extra_config={
                "vocab_size": int(language.config.vocab_size),
                "max_position_embeddings": int(
                    getattr(language.config, "max_position_embeddings", seq_len)
                ),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "max_seq_len": seq_len,
                "batch_size": batch_size,
                "image_token_id": inputs["image_token_id"],
                "seq_len_per_image": inputs["seq_len_per_image"],
            },
            trt_settings={
                **ctx.trt_settings,
                **language_edge_trt_settings(),
            },
        )
    finally:
        restore_attention(patched)

    return {
        "engine_path": engine_path,
        "tensors": {
            "lm_hidden": inputs["lm_hidden"],
        },
        "metadata": {
            "batch_size": batch_size,
            "language_inputs": inputs["language_inputs"],
            "max_seq_len": seq_len,
            "hidden_size": hidden_size,
            "image_token_id": int(inputs["image_token_id"]),
            "seq_len_per_image": int(inputs["seq_len_per_image"]),
        },
    }

def save_artifacts(ctx: EdgeContext, inputs: dict, result: dict) -> None:
    tokenizer = ctx.export_state.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("preprocess must stash tokenizer on ctx.export_state['tokenizer']")

    language_model = inputs["language_model"]
    language_dir = ctx.engine_root / "language"

    save_embedding_table(language_model, language_dir)
    save_tokenizer_for_edge_llm(
        language_dir,
        tokenizer=tokenizer,
        chat_template={
            "image_token_id": inputs["image_token_id"],
            "seq_len_per_image": inputs["seq_len_per_image"],
        },
    )

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    return result