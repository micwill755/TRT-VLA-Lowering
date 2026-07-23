from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.rope import make_rope_rotary_cos_sin
from trt.compile import compile_trt_module
from trt.executor.models.groot.load.serialize import SerializedGrootLanguage
from trt.modules.export.language import CausalLMExportModule
from trt.pipelines.parity import maybe_override_upstream
from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import patch_language_attention, restore_attention
from trt.serialize import SerializedTRTEngine

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    inputs = maybe_override_upstream(ctx, "language", inputs)

    eagle = ctx.model.backbone.eagle_model
    language = eagle.language_model
    language = language.to(device=ctx.device, dtype=ctx.dtype).eval()

    # --- upstream vision output: [B * S_visual, H_lm] ---
    image_embs = inputs["tensors"]["image_embs"]

    # --- tokenized prompt (pipeline-level preprocess) ---
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    image_token_index = getattr(
        eagle, "image_token_index", eagle.config.image_token_index
    )

    # --- build packed language embeddings (splice vision rows into token embeds) ---
    vocab_size = int(language.get_input_embeddings().num_embeddings)
    safe_ids = torch.where(
        input_ids >= vocab_size,
        torch.zeros_like(input_ids),
        input_ids,
    )
    input_embs = language.get_input_embeddings()(safe_ids)
    bsz, seq_len, hidden = input_embs.shape

    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)
    image_token_mask = (flat_ids == image_token_index) | (flat_ids >= vocab_size)

    flat_image_embs = image_embs.reshape(-1, hidden).to(
        device=flat_embs.device,
        dtype=flat_embs.dtype,   # match embedding dtype, not hardcoded fp16
    )
    num_slots = int(image_token_mask.sum().item())
    flat_embs[image_token_mask] = flat_image_embs[:num_slots]
    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden)
    inputs_embeds = inputs_embeds.to(device=ctx.device, dtype=ctx.dtype).contiguous()
    position_ids = torch.arange(seq_len, device=ctx.device, dtype=torch.long).unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=ctx.device)

    # --- EAGER: stock HF path (match model weight dtype, test_vla rung A) ---
    eager_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

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
    ).eval().to(device=ctx.device)

    # --- flat Edge-LLM bindings ---
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        ctx.device,
        language_model=language,
        position_ids=position_ids,
    )

    ctx_len = torch.full((bsz,), seq_len, device=ctx.device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=ctx.device, dtype=torch.int64)

    kv_caches = [
        torch.zeros(
            bsz,
            2,
            num_key_value_heads,
            seq_len,
            head_dim,
            device=ctx.device,
            dtype=ctx.dtype,
        )
        for _ in range(num_layers)
    ]
    kvcache_start_index = torch.empty(0, device=ctx.device, dtype=torch.int32)  # fresh prefill
    ds_stack = torch.zeros(
        0, bsz, seq_len, hidden_size, device=ctx.device, dtype=ctx.dtype
    )

    lm_inputs = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        ds_stack,
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

def compile(ctx: EdgeContext, inputs: dict) -> dict:
    lm_module = inputs["lm_module"]
    lm_inputs = inputs["lm_inputs"]
    decoder = inputs["decoder"]
    hidden_size = inputs["hidden_size"]
    num_attention_heads = inputs["num_attention_heads"]
    num_key_value_heads = inputs["num_key_value_heads"]
    head_dim = inputs["head_dim"]

    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        context_attention_mask_type=ContextAttentionMaskType.CAUSAL,
    )
    try:
        lm_engine = compile_trt_module(
            lm_module,
            lm_inputs,
            {**ctx.trt_settings, "use_python_runtime": True},
        )
    finally:
        restore_attention(patched)

    return {
        "lm_engine": lm_engine,
    }

def load(ctx: EdgeContext, inputs: dict) -> dict:
    serialized_language = SerializedGrootLanguage(
        SerializedTRTEngine(ctx.engine_root / "language")
    )
    return {
        "serialized_engine": serialized_language,
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
    language = inputs["language_model"]
    eager = inputs["eager_inputs"]
    lm_dtype = next(language.parameters()).dtype
    inputs_embeds = eager["inputs_embeds"].to(device=ctx.device, dtype=lm_dtype)

    with torch.no_grad():
        out = language(
            inputs_embeds=inputs_embeds,
            attention_mask=eager["attention_mask"],
            position_ids=eager["position_ids"],
            output_hidden_states=True,
            return_dict=True,
        )

    return {
        "tensors": {
            "logits": out.logits,
            "lm_hidden": out.hidden_states[-1],
            "prefix_k": None,
            "prefix_v": None,
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
    module = inputs["serialized_engine"]

    # SerializedGrootLanguage takes the same positional lm_inputs tuple built in
    # preprocess and returns either logits or (logits, lm_hidden).
    lm_inputs = inputs["lm_inputs"]

    with torch.no_grad():
        out = module(*lm_inputs)

    if isinstance(out, (tuple, list)):
        logits = out[0]
        lm_hidden = out[1] if len(out) > 1 else out[0]
    else:
        logits, lm_hidden = None, out

    return {
        "tensors": {
            "logits": logits,
            "lm_hidden": lm_hidden,
            "prefix_k": None,
            "prefix_v": None,
        },
        "metadata": {
            "backend": "serialized_trt",
        },
    }

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    tensors = result["tensors"]
    ctx.inference.logits = tensors.get("logits")
    ctx.inference.lm_hidden_states = tensors["lm_hidden"]
    ctx.inference.prefix_k = tensors.get("prefix_k")
    ctx.inference.prefix_v = tensors.get("prefix_v")
    return result