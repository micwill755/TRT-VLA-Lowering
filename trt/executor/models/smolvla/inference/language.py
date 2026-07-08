from __future__ import annotations

import torch

from trt.compile import compile_trt_module
from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.executor.models.smolvla.helpers import build_smolvla_prefix_embs
from trt.executor.models.smolvla.load.serialize import SerializedSmolVLALanguage
from trt.modules.export.language import CausalLMExportModule
from trt.pipelines.parity import maybe_override_upstream
from trt.plugin.plugin_utils import patch_language_attention, restore_attention
from trt.prefix_cache import extract_stacked_kv_from_cache
from trt.rope import make_rope_rotary_cos_sin
from trt.serialize import SerializedTRTEngine


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    inputs = maybe_override_upstream(ctx, "language", inputs)

    vlm = ctx.model.vlm_with_expert.get_vlm_model()
    language = vlm.text_model.to(device=ctx.device, dtype=ctx.dtype).eval()
    lm_head = ctx.model.vlm_with_expert.vlm.lm_head

    image_embs = inputs["tensors"]["image_embs"]

    inputs_embeds, prefix_pad_mask, prefix_attention_mask, prefix_position_ids = (
        build_smolvla_prefix_embs(
            ctx.model,
            inputs["img_masks"],
            inputs["tokens"],
            inputs["masks"],
            image_embs,
            inputs["images"],
            inputs["state"],
        )
    )

    bsz, seq_len, hidden = inputs_embeds.shape
    inputs_embeds = inputs_embeds.to(device=ctx.device, dtype=ctx.dtype).contiguous()
    lm_dtype = next(language.parameters()).dtype
    prefix_attention_mask = prefix_attention_mask.to(device=ctx.device, dtype=lm_dtype)

    eager_inputs = {
        "inputs_embeds": inputs_embeds.to(dtype=lm_dtype),
        "attention_mask": prefix_attention_mask,
        "position_ids": prefix_position_ids,
    }

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
    ).eval().to(device=ctx.device)

    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        ctx.device,
        language_model=language,
        position_ids=prefix_position_ids,
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
    kvcache_start_index = torch.empty(0, device=ctx.device, dtype=torch.int32)

    lm_inputs = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    ctx.inference.action_side["prefix_pad_mask"] = prefix_pad_mask.detach()

    return {
        "language_model": language,
        "eager_inputs": eager_inputs,
        "lm_inputs": lm_inputs,
        "lm_module": lm_module,
        "decoder": decoder,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "prefix_pad_mask": prefix_pad_mask,
    }


def compile(ctx: EdgeContext, inputs: dict) -> dict:
    lm_module = inputs["lm_module"]
    lm_inputs = inputs["lm_inputs"]
    decoder = inputs["decoder"]

    patched = patch_language_attention(
        decoder,
        hidden_size=inputs["hidden_size"],
        num_attention_heads=inputs["num_attention_heads"],
        num_key_value_heads=inputs["num_key_value_heads"],
        head_dim=inputs["head_dim"],
        enable_bidirectional_prefill=1,
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
    serialized_language = SerializedSmolVLALanguage(
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

    with torch.no_grad():
        lm_dtype = next(language.parameters()).dtype
        eager_embs = eager["inputs_embeds"].to(device=ctx.device, dtype=lm_dtype)
        out = language(
            inputs_embeds=eager_embs,
            attention_mask=eager["attention_mask"],
            position_ids=eager["position_ids"],
            past_key_values=None,
            use_cache=True,
        )
        lm_hidden = out.last_hidden_state
        prefix_k, prefix_v = extract_stacked_kv_from_cache(
            out.past_key_values,
            num_layers=inputs["num_layers"],
            batch_size=int(eager_embs.shape[0]),
            num_kv_heads=inputs["num_key_value_heads"],
            head_dim=inputs["head_dim"],
            device=ctx.device,
            dtype=ctx.dtype,
        )

    return {
        "tensors": {
            "logits": None,
            "lm_hidden": lm_hidden,
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
        },
        "metadata": {
            "backend": "eager",
            "prefix_pad_mask": inputs["prefix_pad_mask"],
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
            "prefix_pad_mask": inputs["prefix_pad_mask"],
        },
    }


def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    module = inputs["serialized_engine"]
    lm_inputs = inputs["lm_inputs"]

    with torch.no_grad():
        logits, lm_hidden, prefix_k, prefix_v = module(*lm_inputs)

    return {
        "tensors": {
            "logits": logits,
            "lm_hidden": lm_hidden,
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
        },
        "metadata": {
            "backend": "serialized_trt",
            "prefix_pad_mask": inputs["prefix_pad_mask"],
        },
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    tensors = result["tensors"]
    ctx.inference.logits = tensors.get("logits")
    ctx.inference.lm_hidden_states = tensors["lm_hidden"]
    ctx.inference.prefix_k = tensors.get("prefix_k")
    ctx.inference.prefix_v = tensors.get("prefix_v")
    prefix_pad_mask = result.get("metadata", {}).get("prefix_pad_mask")
    if prefix_pad_mask is not None:
        ctx.inference.action_side["prefix_pad_mask"] = prefix_pad_mask
    return result
