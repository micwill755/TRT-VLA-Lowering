"""PI05 language export: compact prefix prefill -> TRT LLMEngineRunner engine."""

from __future__ import annotations

import torch

from trt.compile import save_trt_engine_module
from trt.language import (
    language_edge_llm_config,
    language_edge_trt_settings,
    make_language_edge_input_specs,
)
from trt.modules.export.language import CausalLMExportModule
from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import patch_language_attention, restore_attention
from trt.rope import make_rope_rotary_cos_sin
from trt.context import EdgeContext
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm
from trt.executor.models.pi05.helpers import (
    build_pi05_prefix_embs,
    pad_pi05_compact_prefix,
    pi05_compact_prefix_max_seq_len,
)


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    paligemma = ctx.model.paligemma_with_expert.paligemma.model
    language = paligemma.language_model
    lm_head = ctx.model.paligemma_with_expert.paligemma.lm_head
    device, dtype = ctx.device, ctx.dtype

    image_embs = inputs["tensors"]["image_embs"]

    compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids = (
        build_pi05_prefix_embs(
            ctx.model,
            inputs["img_masks"],
            inputs["tokens"],
            inputs["masks"],
            image_embs,
            inputs["images"],
        )
    )

    tokenizer_max_length = int(getattr(ctx.model.config, "tokenizer_max_length", 200))
    max_seq_len = pi05_compact_prefix_max_seq_len(
        image_embs,
        inputs["images"],
        tokenizer_max_length,
    )
    inputs_embeds, prefix_pad_mask, prefix_attention_mask, prefix_position_ids, valid_seq_len = (
        pad_pi05_compact_prefix(
            compact_embs,
            compact_pad_mask,
            compact_attention_mask,
            compact_position_ids,
            max_seq_len=max_seq_len,
        )
    )

    bsz, seq_len, hidden = inputs_embeds.shape
    trt_inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()

    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))

    decoder = getattr(language, "model", language)
    num_layers = len(decoder.layers)

    lm_module = CausalLMExportModule(
        decoder,
        lm_head,
        select_layer=-1,
    ).eval().to(device=device)

    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        device,
        language_model=language,
        position_ids=prefix_position_ids,
    )

    ctx_len = torch.full((bsz,), valid_seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), valid_seq_len - 1, device=device, dtype=torch.int64)

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
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)

    lm_inputs = (
        trt_inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    lm_hidden = torch.zeros(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    return {
        "language_model": language,
        "lm_inputs": lm_inputs,
        "lm_module": lm_module,
        "decoder": decoder,
        "lm_hidden": lm_hidden,
        "batch_size": bsz,
        "seq_len": seq_len,
        "valid_seq_len": valid_seq_len,
        "max_seq_len": max_seq_len,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "prefix_pad_mask": prefix_pad_mask,
        "language_inputs": {
            "inputs_embeds": trt_inputs_embeds,
            "attention_mask": prefix_attention_mask,
            "position_ids": prefix_position_ids,
        },
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
    input_specs = make_language_edge_input_specs(
        input_names,
        lm_inputs,
        batch_size=batch_size,
        max_seq_len=seq_len,
        static_prefill_seq_len=True,
    )

    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        context_attention_mask_type=ContextAttentionMaskType.PADDING,
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
            extra_config={
                **language_edge_llm_config(
                    language.config,
                    max_seq_len=seq_len,
                    batch_size=batch_size,
                    num_layers=num_hidden_layers,
                ),
                "context_attention_mask_type": int(ContextAttentionMaskType.PADDING),
            },
            input_specs=input_specs,
            flat_tensors=lm_inputs,
            trt_settings={
                **ctx.trt_settings,
                **language_edge_trt_settings(),
            },
        )
    finally:
        restore_attention(patched)

    batch_size = int(inputs["batch_size"])
    seq_len = int(inputs["seq_len"])
    num_hidden_layers = int(inputs["num_hidden_layers"])
    num_key_value_heads = int(inputs["num_key_value_heads"])
    head_dim = int(inputs["head_dim"])

    # Dummy prefix KV for the downstream diffusion trace (shape/dtype only).
    prefix_k = torch.zeros(
        num_hidden_layers,
        batch_size,
        num_key_value_heads,
        seq_len,
        head_dim,
        device=ctx.device,
        dtype=ctx.dtype,
    )
    prefix_v = torch.zeros_like(prefix_k)

    return {
        "engine_path": engine_path,
        "tensors": {
            "lm_hidden": inputs["lm_hidden"],
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
        },
        "metadata": {
            "batch_size": batch_size,
            "language_inputs": inputs["language_inputs"],
            "max_seq_len": int(inputs["max_seq_len"]),
            "valid_seq_len": int(inputs["valid_seq_len"]),
            "hidden_size": int(inputs["hidden_size"]),
            "prefix_pad_mask": inputs["prefix_pad_mask"],
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
            "max_seq_len": int(inputs["max_seq_len"]),
        },
    )


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    return result
