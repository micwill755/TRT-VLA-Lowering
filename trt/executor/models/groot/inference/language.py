from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.rope import make_rope_rotary_cos_sin
from trt.compile import compile_trt_module
from trt.modules.export.language import CausalLMExportModule, ContextProjectionExportModule

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    eagle = ctx.model.backbone.eagle_model
    language = eagle.language_model
    device, dtype = ctx.device, ctx.dtype

    # --- upstream vision output: [B * S_visual, H_lm] ---
    image_embs = inputs["tensors"]["image_embs"]

    # --- tokenized prompt ---
    input_ids = ctx.inference.tokenized["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = ctx.inference.tokenized.get("attention_mask")

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

    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden)
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

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
        inputs_embeds,          # spliced vision rows — NOT raw input_embs
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    # --- patch -> compile -> restore ---
    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        enable_bidirectional_prefill=0,
    )
    try:
        lm_engine = compile_trt_module(
            lm_module, lm_inputs, {**ctx.trt_settings, "use_python_runtime": True}
        )
    finally:
        restore_attention(patched)

    return {
        "lm_inputs": lm_inputs,
        "lm_module": lm_module,
        "lm_engine": lm_engine,
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
    lm_module = inputs["lm_module"]
    lm_inputs = inputs["lm_inputs"]

    with torch.no_grad():
        logits, lm_hidden, prefix_k, prefix_v = lm_module(*lm_inputs)

    return {
        "tensors": {
            "logits": logits,
            "lm_hidden": lm_hidden,
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
        },
        "metadata": {
            "backend": "eager",
        },
    }

def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    lm_engine = inputs["lm_engine"]
    lm_inputs = inputs["lm_inputs"]

    with torch.no_grad():
        logits, lm_hidden, prefix_k, prefix_v = lm_engine(*lm_inputs)

    return {
        "tensors": {
            "logits": logits,
            "lm_hidden": lm_hidden,
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
    ctx.inference.lm_hidden = tensors["lm_hidden"]
    ctx.inference.prefix_k = tensors.get("prefix_k")
    ctx.inference.prefix_v = tensors.get("prefix_v")
    return result