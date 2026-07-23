from __future__ import annotations

import torch

from trt.compile import compile_trt_module
from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.executor.models.smolvla.helpers import build_smolvla_prefix_embs
from trt.executor.models.smolvla.load.serialize import SerializedSmolVLALanguage
from trt.modules.export.language import CausalLMExportModule
from trt.pipelines.parity import maybe_override_upstream
from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import patch_language_attention, restore_attention
from trt.prefix_cache import PrefixKVCache
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
        context_attention_mask_type=ContextAttentionMaskType.PADDING,
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
    num_layers = inputs["num_layers"]
    num_kv_heads = inputs["num_key_value_heads"]
    head_dim = inputs["head_dim"]
    device = ctx.device
    dtype = ctx.dtype

    with torch.no_grad():
        lm_dtype = next(language.parameters()).dtype
        eager_embs = eager["inputs_embeds"].to(device=device, dtype=lm_dtype)
        batch_size = int(eager_embs.shape[0])
        out = language(
            inputs_embeds=eager_embs,
            attention_mask=eager["attention_mask"],
            position_ids=eager["position_ids"],
            past_key_values=None,
            use_cache=True,
        )
        lm_hidden = out.last_hidden_state
        past_key_values = out.past_key_values

        # TODO: need to improve this
        if isinstance(past_key_values, PrefixKVCache):
            prefix_k = past_key_values.key_cache.to(device=device, dtype=dtype)
            prefix_v = past_key_values.value_cache.to(device=device, dtype=dtype)
        elif isinstance(past_key_values, tuple) and len(past_key_values) == 2:
            if past_key_values[0] is None or past_key_values[1] is None:
                prefix_k = torch.zeros(
                    num_layers, batch_size, num_kv_heads, 0, head_dim, dtype=dtype, device=device
                )
                prefix_v = torch.zeros_like(prefix_k)
            else:
                prefix_k = past_key_values[0].to(device=device, dtype=dtype)
                prefix_v = past_key_values[1].to(device=device, dtype=dtype)
        elif past_key_values is None or (
            hasattr(past_key_values, "layers") and len(past_key_values.layers) == 0
        ):
            prefix_k = torch.zeros(
                num_layers, batch_size, num_kv_heads, 0, head_dim, dtype=dtype, device=device
            )
            prefix_v = torch.zeros_like(prefix_k)
        elif hasattr(past_key_values, "layers"):
            layers = list(past_key_values.layers)
            layer_k: list[torch.Tensor | None] = []
            layer_v: list[torch.Tensor | None] = []
            any_initialized = False
            for idx in range(num_layers):
                if idx >= len(layers):
                    layer_k.append(None)
                    layer_v.append(None)
                    continue
                k = getattr(layers[idx], "keys", None)
                v = getattr(layers[idx], "values", None)
                if (k is None) != (v is None):
                    raise ValueError(
                        f"Inconsistent cache state at layer {idx}: one of keys/values is None"
                    )
                if k is not None:
                    any_initialized = True
                layer_k.append(k)
                layer_v.append(v)

            if not any_initialized:
                prefix_k = torch.zeros(
                    num_layers, batch_size, num_kv_heads, 0, head_dim, dtype=dtype, device=device
                )
                prefix_v = torch.zeros_like(prefix_k)
            else:
                first_k = next(k for k in layer_k if k is not None)
                prefix_len = int(first_k.shape[-2])
                filled_k: list[torch.Tensor] = []
                filled_v: list[torch.Tensor] = []
                for idx, (k, v) in enumerate(zip(layer_k, layer_v)):
                    if k is None:
                        k = torch.zeros(
                            batch_size,
                            num_kv_heads,
                            prefix_len,
                            head_dim,
                            dtype=dtype,
                            device=device,
                        )
                        v = torch.zeros_like(k)
                    else:
                        if int(k.shape[-2]) != prefix_len:
                            raise ValueError(
                                "Cache sequence length mismatch across layers; "
                                f"layer0={prefix_len}, layer{idx}={int(k.shape[-2])}"
                            )
                        k = k.to(device=device, dtype=dtype)
                        v = v.to(device=device, dtype=dtype)
                    filled_k.append(k)
                    filled_v.append(v)
                prefix_k = torch.stack(filled_k, dim=0)
                prefix_v = torch.stack(filled_v, dim=0)
        else:
            raise ValueError(
                "past_key_values must be None, (prefix_k, prefix_v), PrefixKVCache, "
                "or an object with .layers"
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
