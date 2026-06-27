"""Shared language prefill wiring for Edge-LLM LM engines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trt.inference.context import LanguageOutputs
from trt.io_spec import ComponentIOSpec
from trt.rope import make_rope_rotary_cos_sin


@dataclass
class LanguagePrefillInputs:
    inputs_embeds: torch.Tensor
    rope_rotary_cos_sin: torch.Tensor
    context_lengths: torch.Tensor
    kvcache_start_index: torch.Tensor
    last_token_ids: torch.Tensor
    kv_caches: tuple[torch.Tensor, ...]


def build_language_prefill_inputs(
    language_inputs: dict,
    *,
    language_model: nn.Module,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> LanguagePrefillInputs:
    """Build RoPE, KV caches, and prefill controls matching ``LLMEngineRunner``."""
    lm_inputs = language_inputs["inputs_embeds"].to(device=device, dtype=dtype)
    batch_size = int(lm_inputs.shape[0])
    seq_len = int(lm_inputs.shape[1])
    max_seq_len = max(int(max_seq_len), seq_len)

    kv_caches = tuple(
        torch.zeros(
            batch_size,
            2,
            int(num_key_value_heads),
            max_seq_len,
            int(head_dim),
            device=device,
            dtype=dtype,
        )
        for _ in range(int(num_layers))
    )
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        language_model.config,
        max_seq_len,
        device,
        language_model=language_model,
        position_ids=language_inputs.get("position_ids"),
    )
    return LanguagePrefillInputs(
        inputs_embeds=lm_inputs,
        rope_rotary_cos_sin=rope_rotary_cos_sin,
        context_lengths=torch.full((batch_size,), seq_len, device=device, dtype=torch.int32),
        kvcache_start_index=torch.empty(0, dtype=torch.int32, device=device),
        last_token_ids=torch.full((batch_size, 1), seq_len - 1, device=device, dtype=torch.int64),
        kv_caches=kv_caches,
    )


def run_language_prefill(
    module,
    prefill: LanguagePrefillInputs,
    io: ComponentIOSpec,
) -> LanguageOutputs:
    base_args = (
        prefill.inputs_embeds,
        prefill.rope_rotary_cos_sin,
        prefill.context_lengths,
        prefill.kvcache_start_index,
        prefill.last_token_ids,
    )
    if getattr(module, "bundles_kv_caches", False):
        outputs = module(*base_args, prefill.kv_caches)
    else:
        outputs = module(*base_args, *prefill.kv_caches)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    return LanguageOutputs.from_tuple(outputs, io)
