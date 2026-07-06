from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.rope import make_rope_rotary_cos_sin

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    eagle = ctx.model.backbone.eagle_model
    language = eagle.language_model

    input_ids = ctx.inference.tokenized["input_ids"].to(ctx.device)
    attention_mask = ctx.inference.tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(ctx.device)

    # From upstream vision stage.
    image_embs = inputs["tensors"]["image_embs"]

    image_token_index = getattr(eagle, "image_token_index", eagle.config.image_token_index)

    token_embs = language.get_input_embeddings()(input_ids)
    bsz, seq_len, hidden = token_embs.shape

    flat_embs = token_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)
    image_token_mask = flat_ids == image_token_index

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

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=ctx.device)

    position_ids = torch.arange(seq_len, device=ctx.device, dtype=torch.long).unsqueeze(0)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_token_mask": image_token_mask.reshape(bsz, seq_len),
    }

def execute(ctx: EdgeContext, inputs: dict) -> dict:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx, inputs)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx, inputs)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx, inputs)

    raise ValueError(f"unsupported execution mode: {ctx.execution_mode}")

def _run_eager(ctx: EdgeContext, inputs: dict) -> dict:
    language = ctx.model.backbone.eagle_model.language_model

    # Match the parity path you validated: fp16 + final hidden.
    inputs_embeds = inputs["inputs_embeds"].to(device=ctx.device, dtype=ctx.dtype)
    attention_mask = inputs["attention_mask"].to(ctx.device)
    position_ids = inputs["position_ids"].to(ctx.device)

    with torch.no_grad():
        out = language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    return {
        "tensors": {
            "lm_hidden_states": out.hidden_states[-1],
            "logits": out.logits,
        },
        "metadata": {
            "backend": "eager",
        },
    }


def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    module = ctx.handles.in_memory.language
    if module is None:
        raise RuntimeError("in-memory TRT backend missing language module")

    flat_tensors = _build_language_flat_tensors(ctx, inputs)

    with torch.no_grad():
        out = module(*flat_tensors)

    logits, lm_hidden_states, prefix_k, prefix_v = (
            out["logits"],
            out["lm_hidden_states"],
            out.get("prefix_k"),
            out.get("prefix_v"),
        )

    return {
        "tensors": {
            "logits": logits,
            "lm_hidden_states": lm_hidden_states,
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
        },
        "metadata": {
            "backend": "in_memory_trt",
        },
    }

def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    module = ctx.handles.serialized.language
    if module is None:
        raise RuntimeError("serialized TRT backend missing language module")

    flat_tensors = _build_language_flat_tensors(ctx, inputs)

    with torch.no_grad():
        out = module(*flat_tensors)

    logits, lm_hidden_states, prefix_k, prefix_v = _unpack_language_outputs(out)

    return {
        "tensors": {
            "logits": logits,
            "lm_hidden_states": lm_hidden_states,
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
        },
        "metadata": {
            "backend": "serialized_trt",
        },
    }

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    tensors = result["tensors"]

    ctx.inference.logits = tensors.get("logits")
    ctx.inference.lm_hidden_states = tensors["lm_hidden_states"]
    ctx.inference.prefix_k = tensors.get("prefix_k")
    ctx.inference.prefix_v = tensors.get("prefix_v")

    return result

def _build_language_flat_tensors(ctx: EdgeContext, inputs: dict) -> tuple[torch.Tensor, ...]:
    """Build the flat input tuple expected by the Edge-LLM language engine.

    HF eager only needs ``language(inputs_embeds, attention_mask, position_ids)``.
    The TRT/plugin path uses the C++ LLMEngineRunner ABI consumed by
    ``CausalLMExportModule.forward()``:

    ``(inputs_embeds, rope, ctx_len, kvcache_start_index, last_token_ids, *kv_caches)``
    """

    # Qwen3ForCausalLM wrapper on the Eagle backbone.
    language = ctx.model.backbone.eagle_model.language_model
    # Inner Qwen3Model that owns ``.layers`` (unwrap ForCausalLM -> model).
    decoder = getattr(language, "model", language)
    # Model config: hidden_size, num_attention_heads, num_key_value_heads, head_dim.
    cfg = language.config

    # Packed prompt embeddings from preprocess(): text rows with vision spliced in.
    # Shape: [B, S, H], e.g. [1, 566, 2048]. fp16 + contiguous for plugin/TRT runtime.
    inputs_embeds = inputs["inputs_embeds"].to(
        device=ctx.device,
        dtype=torch.float16,
    ).contiguous()

    # bsz: batch size (usually 1). seq_len: packed language length after image expansion.
    bsz, seq_len, _ = inputs_embeds.shape
    # Per-head dimension for attention and RoPE (Qwen3: cfg.head_dim, typically 128).
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    # GQA KV head count (may be smaller than query head count).
    num_key_value_heads = int(cfg.num_key_value_heads)
    # One KV cache tensor per decoder layer (e.g. 12 for GR00T language).
    num_layers = len(decoder.layers)

    # Token positions for RoPE lookup. Shape: [B, S], e.g. [[0, 1, ..., 565]].
    position_ids = inputs["position_ids"].to(ctx.device)

    # External RoPE table for PluginAttention (HF eager builds this internally).
    # Shape: [1, max_seq_len, rotary_dim], e.g. [1, 566, 128]. Must be fp32.
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        ctx.device,
        language_model=language,
        position_ids=position_ids,
    )

    # Per-batch valid sequence length for this prefill. Shape: [B], e.g. [566].
    # sum(attention_mask) handles padding; all-ones mask gives seq_len. int32 for plugin API.
    ctx_len = inputs["attention_mask"].sum(dim=1).to(
        device=ctx.device,
        dtype=torch.int32,
    )

    # KV cache write start index. Shape [0] means fresh prefill from position 0.
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=ctx.device)
    # Index of last valid token for lm_head gather. Shape: [B, 1], e.g. [[565]].
    last_token_ids = (ctx_len - 1).reshape(bsz, 1).to(dtype=torch.int64)

    # One KV buffer per layer, passed as *past_key_values to CausalLMExportModule.
    # Per-layer shape: [B, 2, num_key_value_heads, seq_len, head_dim]
    #   B=2 -> key + value; seq_len is cache capacity for this profile.
    kv_caches = [
        torch.zeros(
            bsz,
            2,
            num_key_value_heads,
            seq_len,
            head_dim,
            device=ctx.device,
            dtype=torch.float16,
        )
        for _ in range(num_layers)
    ]

    # Tuple order must match CausalLMExportModule.forward positional args exactly.
    return (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )