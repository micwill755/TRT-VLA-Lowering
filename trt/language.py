"""Plugin-attention language models for TensorRT export and Edge-LLM runtime.

This module wraps HuggingFace decoder stacks with ``PluginAttention`` (trt attention plugin)
and manual layer loops so graphs match ``LLMEngineRunner`` I/O:

  inputs_embeds, rope_rotary_cos_sin, context_lengths, kvcache_start_index,
  last_token_ids, past_key_values_*  ->  logits, lm_hidden_states, prefix_k, prefix_v

KV cache updates happen in-place on ``past_key_values_*`` inputs; prefix K/V are
stacked views exported for the action head.
"""

import copy
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt

from trt.attention import PluginAttention
from trt.compile import compile_trt_module, save_trt_engine_module
from trt.io_spec import ComponentIOSpec, VLA_LANGUAGE_IO
from trt.rope import (
    config_to_dict,
    export_rope_fields,
    language_head_dim,
    make_dummy_rope_rotary_cos_sin,
    make_rope_rotary_cos_sin,
)
from trt.utils import free_cuda_memory

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE_TRT_SETTINGS: dict[str, Any] = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
    "assume_dynamic_shape_support": True,
}


@dataclass
class LanguageEngineSpec:
    """Model-specific language export configuration for ``save_language_engine_for_edge_llm``."""

    decoder: nn.Module
    lm_head: nn.Module
    language_model: nn.Module
    config: Any

    prefix_embs: torch.Tensor
    batch_size: int
    max_seq_len: int
    hidden_size: int
    num_layers: int
    num_key_value_heads: int
    head_dim: int

    image_token_id: int
    seq_len_per_image: int = 0

    position_ids: torch.Tensor | None = None
    select_layer: int = -1
    enable_bidirectional_prefill: int = 1
    static_prefill_seq_len: bool = False
    context_hidden_size: int | None = None
    export_dtype: torch.dtype = torch.float16

    io: ComponentIOSpec = VLA_LANGUAGE_IO
    trt_settings: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_LANGUAGE_TRT_SETTINGS)
    )
    model_type: str = "language"
    log_prefix: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _as_tensor(x):
    """Unwrap tuple/list outputs from patched attention modules."""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def gather_last_token_hidden(
    hidden_states: torch.Tensor,
    last_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, S, H] at last_token_ids [B] or [B, 1] -> [B, H] for lm_head."""
    if last_token_ids.ndim == 1:
        indices = last_token_ids
    else:
        indices = last_token_ids.squeeze(-1)
    batch_idx = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.long,
    )
    return hidden_states[batch_idx, indices]


def _install_plugin_attention(
    lm: nn.Module,
    config,
    enable_bidirectional_prefill: int = 1,
) -> None:
    """Replace each decoder layer's self_attn with PluginAttention."""
    from trt.plugin_utils import set_plugin_config_from_model

    set_plugin_config_from_model(
        config,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
    )
    for i, layer in enumerate(lm.layers):
        layer.self_attn = PluginAttention(
            layer.self_attn,
            config,
            layer_idx=i,
            enable_bidirectional_prefill=enable_bidirectional_prefill,
        ).eval()


# ---------------------------------------------------------------------------
# Plugin LM cores
# ---------------------------------------------------------------------------

class PluginLMForCausalLM(nn.Module):
    """Edge-LLM causal LM: manual decoder loop -> logits + lm_hidden_states + prefix KV.

    External RoPE and KV-cache controls match ``LLMEngineRunner`` prefill/decode.
    ``select_layer=-1`` (default for GR00T TRT) uses final RMSNorm hidden for
    context; positive values capture an intermediate layer output pre-norm.
    """

    def __init__(
        self,
        lm: nn.Module,
        lm_head: nn.Module,
        *,
        select_layer: int = -1,
    ):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.select_layer = int(select_layer)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        lm_dtype = next(self.lm.parameters()).dtype
        hidden = _as_tensor(inputs_embeds).to(dtype=lm_dtype)
        context_hidden = hidden if self.select_layer == 0 else None
        new_kvs = []

        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                past_key_value=past_key_values[i],
                ctx_len=context_lengths,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = _as_tensor(hidden)
            hidden = residual + hidden

            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = _as_tensor(layer.mlp(hidden))
            hidden = residual + hidden
            new_kvs.append(kv)

            if self.select_layer > 0 and (i + 1) == self.select_layer:
                context_hidden = hidden

        hidden = _as_tensor(self.lm.norm(hidden))
        if context_hidden is None:
            context_hidden = hidden

        last_hidden = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last_hidden).float()

        seq_len = inputs_embeds.shape[1]
        prefix_k = torch.stack(
            [kv[:, 0, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        prefix_v = torch.stack(
            [kv[:, 1, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        return logits, context_hidden, prefix_k, prefix_v


# ---------------------------------------------------------------------------
# GR00T: causal LM + context projection (dual outputs)
# ---------------------------------------------------------------------------

class GROOTContextProjectionWrapper(nn.Module):
    """eagle_linear -> vlln -> vl_self_attention (matches eager context path)."""

    def __init__(self, eagle_linear, vlln, vl_self_attention):
        super().__init__()
        self.eagle_linear = eagle_linear
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, hidden_states: torch.Tensor):
        context_embs = self.eagle_linear(hidden_states)

        vlln_weight = getattr(self.vlln, "weight", None)
        if vlln_weight is not None:
            context_embs = context_embs.to(dtype=vlln_weight.dtype)

        context_embs = self.vlln(context_embs)
        context_embs = self.vl_self_attention(context_embs)
        return context_embs


class GROOTLanguageContextWrapper(nn.Module):
    """Legacy fused export: causal LM logits + GR00T context_embs."""

    def __init__(
        self,
        lm_wrapper: PluginLMForCausalLM,
        context_projection: GROOTContextProjectionWrapper,
    ):
        super().__init__()
        self.lm_wrapper = lm_wrapper
        self.context_projection = context_projection

    def forward(
        self,
        inputs_embeds,
        rope_rotary_cos_sin,
        context_lengths,
        kvcache_start_index,
        last_token_ids,
        *past_key_values,
    ):
        logits, context_hidden = self.lm_wrapper(
            inputs_embeds,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            last_token_ids,
            *past_key_values,
        )[:2]
        context_embs = self.context_projection(context_hidden)
        return logits, context_embs


# ---------------------------------------------------------------------------
# Compile helper (in-memory TRT module)
# ---------------------------------------------------------------------------

def compile_language_trt_with_plugin(
    plugin_language: nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    device: torch.device,
    settings,
    max_seq_len: int | None = None,
    dtype: torch.dtype = torch.float16,
    include_last_token_ids: bool = True,
):
    """Trace and compile a plugin language module with representative prefill inputs."""
    ctx_seq_len = int(inputs_embeds.shape[1])
    max_seq_len = ctx_seq_len if max_seq_len is None else int(max_seq_len)
    batch_size = int(inputs_embeds.shape[0])

    kv_caches = [
        torch.zeros(
            batch_size,
            2,  # key + value
            int(num_key_value_heads),
            max_seq_len,
            int(head_dim),
            device=device,
            dtype=dtype,
        )
        for _ in range(int(num_layers))
    ]

    # Placeholder RoPE for export tracing; runtime fills real values via LLMEngineRunner.
    rope_rotary_cos_sin = torch.randn(
        1,
        int(max_seq_len),
        int(head_dim),
        dtype=torch.float32,
        device=device,
    )

    ctx_len = torch.full(
        (batch_size,),
        ctx_seq_len,
        device=device,
        dtype=torch.int32,
    )
    # Fresh prefill: empty kvcache_start_index, gather logits at last valid token.
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    compile_inputs: tuple[torch.Tensor, ...] = (
        inputs_embeds.to(device=device, dtype=dtype),
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
    )
    if include_last_token_ids:
        last_token_ids = torch.full(
            (batch_size, 1),
            int(ctx_seq_len) - 1,
            device=device,
            dtype=torch.int64,
        )
        compile_inputs = compile_inputs + (last_token_ids,)

    trt_language = compile_trt_module(
        plugin_language,
        compile_inputs + tuple(kv_caches),
        settings,
    )

    return trt_language, max_seq_len


# ---------------------------------------------------------------------------
# Eager reference + PI0.5 compile smoke checks
# ---------------------------------------------------------------------------

def _smoke_first_bad_index(mask, shape):
    flat_idx = int(mask.flatten().nonzero(as_tuple=False)[0].item())
    coords = []
    for dim in reversed(shape):
        coords.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(coords))


def _smoke_tensor_health(name, tensor):
    finite = torch.isfinite(tensor)
    bad = ~finite
    bad_count = int(bad.sum().item())
    if bad_count == 0:
        return

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    first_idx = _smoke_first_bad_index(bad, tensor.shape)
    first_val = tensor[first_idx].detach().cpu().item()
    print(f"{name} nonfinite count:", bad_count, "of", tensor.numel())
    print(f"{name} nan count:", nan_count)
    print(f"{name} inf count:", inf_count)
    print(f"{name} first nonfinite index:", first_idx, "value:", first_val)


def _smoke_error_metrics(name, trt_tensor, eager_tensor, include_top1=False):
    _smoke_tensor_health(f"{name} TRT", trt_tensor)
    _smoke_tensor_health(f"{name} eager", eager_tensor)
    trt_f = trt_tensor.float()
    eager_f = eager_tensor.float()
    diff = trt_f - eager_f
    abs_diff = diff.abs()
    rel_l2 = diff.norm() / eager_f.norm().clamp_min(1e-8)
    rel_mean_pct = abs_diff.mean() / eager_f.abs().mean().clamp_min(1e-8) * 100
    if include_top1:
        top1_match = (trt_tensor.argmax(dim=-1) == eager_tensor.argmax(dim=-1)).float().mean()
    else:
        top1_match = None

    print(f"{name} mean diff:", abs_diff.mean().item())
    print(f"{name} max diff:", abs_diff.max().item())
    print(f"{name} relative L2:", rel_l2.item())
    print(f"{name} relative mean %:", rel_mean_pct.item())
    if top1_match is not None:
        print(f"{name} top1 match %:", (top1_match * 100).item())

@torch.no_grad()
def run_prefix_language_eager(
    language_model: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager HF language prefill for PI0.5 parity: hidden + stacked prefix K/V."""
    from trt.prefix_cache import stack_prefix_kv_from_cache

    device = next(language_model.parameters()).device
    lm_dtype = next(language_model.parameters()).dtype
    eager_embs = inputs_embeds.to(device=device, dtype=lm_dtype)
    out = language_model(
        inputs_embeds=eager_embs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
    )
    hidden = _as_tensor(out.last_hidden_state)
    prefix_k, prefix_v = stack_prefix_kv_from_cache(out.past_key_values)
    return hidden, prefix_k, prefix_v


@torch.no_grad()
def pi05_plugin_lm_smoke_check(
    core,
    trt_language,
    prefix_embs,
    *,
    max_seq_len,
    device,
    attention_mask=None,
    position_ids=None,
    prefix_pad_masks=None,
    max_logit_tokens=16,
):
    """Compare TRT plugin LM hidden/logits/KV against eager HF forward (PI0.5 compile)."""
    lm = core.paligemma_with_expert.paligemma.model.language_model
    lm_head = getattr(core.paligemma_with_expert.paligemma, "lm_head", None)
    if lm_head is None:
        print("LM plugin smoke-check logits: skipped, no lm_head on PaliGemma model")
        return

    lm_dtype = next(lm.parameters()).dtype
    head_dtype = next(lm_head.parameters()).dtype

    eager_prefix_embs = prefix_embs.to(device=device, dtype=lm_dtype)
    eager_out = lm(
        inputs_embeds=eager_prefix_embs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
    )
    eager_hidden = _as_tensor(eager_out.last_hidden_state)

    trt_prefix_embs = prefix_embs.to(device=device, dtype=torch.float16)
    cfg = lm.config
    kv_caches = [
        torch.zeros(
            int(trt_prefix_embs.shape[0]),
            2,  # key + value
            int(cfg.num_key_value_heads),
            max_seq_len,
            int(getattr(
                cfg,
                "head_dim",
                cfg.hidden_size // cfg.num_attention_heads,
            )),
            device=device,
            dtype=trt_prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]
    ctx_len = torch.full(
        (trt_prefix_embs.shape[0],),
        trt_prefix_embs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        max_seq_len,
        device,
        language_model=lm,
        position_ids=position_ids,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (trt_prefix_embs.shape[0], 1),
        trt_prefix_embs.shape[1] - 1,
        device=device,
        dtype=torch.int64,
    )
    trt_logits, trt_hidden, trt_k, trt_v = unpack_vla_prefix_language_outputs(
        trt_language(
            trt_prefix_embs,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            last_token_ids,
            *kv_caches,
        )
    )

    _smoke_tensor_health("LM plugin smoke-check eager hidden", eager_hidden)
    _smoke_tensor_health("LM plugin smoke-check TRT hidden", trt_hidden)

    eager_last_hidden = gather_last_token_hidden(eager_hidden, last_token_ids)
    eager_logits = lm_head(eager_last_hidden.to(device=device, dtype=head_dtype))

    print("LM plugin smoke-check logits shape:", tuple(trt_logits.shape))
    _smoke_error_metrics("LM plugin smoke-check logits", trt_logits, eager_logits, include_top1=True)

    cache = eager_out.past_key_values
    eager_k = torch.stack([layer.keys for layer in cache.layers], dim=0)
    eager_v = torch.stack([layer.values for layer in cache.layers], dim=0)

    _smoke_error_metrics("LM plugin smoke-check prefix_k", trt_k, eager_k)
    _smoke_error_metrics("LM plugin smoke-check prefix_v", trt_v, eager_v)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def unpack_vla_prefix_language_outputs(
    outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split VLA language outputs: logits, lm_hidden_states, prefix_k, prefix_v."""
    return outputs[0], outputs[1], outputs[2], outputs[3]


def make_plugin_lm_causal_wrapper(
    decoder: nn.Module,
    config,
    lm_head: nn.Module,
    *,
    select_layer: int = -1,
    enable_bidirectional_prefill: int = 1,
    log_prefix: str = "",
) -> PluginLMForCausalLM:
    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )

    prefix = f"{log_prefix} " if log_prefix else ""
    print(f"{prefix}head_dim:", head_dim)
    print(f"{prefix}select_layer:", int(select_layer))

    _install_plugin_attention(
        decoder,
        config,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
    )

    return PluginLMForCausalLM(
        decoder,
        lm_head,
        select_layer=select_layer,
    ).eval()


def make_groot_action_context_module(
    core,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> GROOTContextProjectionWrapper:
    """Build GR00T action_context engine (eagle_linear -> vlln -> vl_self_attention)."""
    return GROOTContextProjectionWrapper(
        copy.deepcopy(core.backbone.eagle_linear).to(device=device, dtype=dtype).eval(),
        copy.deepcopy(core.action_head.vlln).to(device=device, dtype=dtype).eval(),
        copy.deepcopy(core.action_head.vl_self_attention).to(device=device, dtype=dtype).eval(),
    ).eval()


def make_groot_language_context_wrapper(
    core,
    decoder: nn.Module,
    config,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    enable_bidirectional_prefill: int = 0,
    select_layer: int | None = None,
) -> GROOTLanguageContextWrapper:
    """Build legacy fused GR00T language engine (logits + context_embs)."""
    if select_layer is None:
        # Eager GR00T reads hidden_states[backbone.select_layer] (pre-norm), but the
        # plugin-attention manual decoder only matches that pipeline after final RMSNorm.
        select_layer = -1

    language_model = core.backbone.eagle_model.language_model
    lm_head = copy.deepcopy(language_model.lm_head).to(device=device, dtype=dtype).eval()
    causal_lm = make_plugin_lm_causal_wrapper(
        decoder,
        config,
        lm_head,
        select_layer=select_layer,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
        log_prefix="groot",
    )
    context = make_groot_action_context_module(core, device=device, dtype=dtype)
    return GROOTLanguageContextWrapper(causal_lm, context).eval()


# ---------------------------------------------------------------------------
# VitRunner-compatible token expansion / embedding export
# ---------------------------------------------------------------------------

def compute_vit_expanded_seq_len(
    input_ids: torch.Tensor,
    image_token_id: int,
    seq_len_per_image: int,
) -> int:
    """Sequence length after VitRunner::textPreprocess placeholder expansion.

    Handles both a single ``image_token_id`` placeholder per image (expanded to
    ``seq_len_per_image`` slots) and prompts that already contain full image-token
    runs (e.g. Eagle chat templates that repeat the image token in the string).
    """
    flat = input_ids.reshape(-1).tolist()
    num_image_tokens = sum(1 for token_id in flat if token_id == image_token_id)
    num_placeholders = 0
    for idx, token_id in enumerate(flat):
        if token_id == image_token_id and (idx == 0 or flat[idx - 1] != image_token_id):
            num_placeholders += 1
    return len(flat) - num_image_tokens + num_placeholders * int(seq_len_per_image)


def make_dummy_inputs_embeds(
    batch_size: int,
    max_seq_len: int,
    hidden_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Random inputs_embeds for TRT trace when runtime performs embedding lookup."""
    return torch.randn(
        int(batch_size),
        int(max_seq_len),
        int(hidden_size),
        device=device,
        dtype=dtype,
    )

# ---------------------------------------------------------------------------
# Edge-LLM TRT input specs (PR #4325 multi-profile)
# ---------------------------------------------------------------------------

def language_edge_output_names(
    language_output_names: tuple[str, ...] | list[str],
    num_kv_layers: int | None = None,
) -> list[str]:
    """Language engine pipeline output bindings (excludes in-place past_key_values I/O)."""
    del num_kv_layers
    return list(language_output_names)


def language_edge_trace_seq_len(max_seq_len: int) -> int:
    """Prefill trace length; matches ``make_language_edge_input_specs`` opt profile."""
    return max(int(max_seq_len) // 2, 1)


def make_language_edge_flat_tensors(
    prefix_embs: torch.Tensor,
    *,
    batch_size: int,
    max_seq_len: int,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    rope_rotary_cos_sin: torch.Tensor | None = None,
    static_prefill_seq_len: bool = False,
) -> tuple[tuple[torch.Tensor, ...], int]:
    """Flat inputs for Edge-LLM language export/trace.

    Traces at the prefill profile opt sequence length so dynamo does not pin
    ``inputs_embeds`` to the profile max (which violates the shared ``seq_len``
    dynamic symbol). KV caches remain sized to ``max_seq_len``.

    Set ``static_prefill_seq_len=True`` for fixed-prefix VLA exports where
    ``prefix_k`` / ``prefix_v`` must match the full prompt length at runtime.
    """
    max_seq_len = int(max_seq_len)
    trace_seq_len = (
        max_seq_len
        if static_prefill_seq_len
        else language_edge_trace_seq_len(max_seq_len)
    )
    trace_embs = prefix_embs[:, :trace_seq_len].to(device=device, dtype=dtype).contiguous()

    if rope_rotary_cos_sin is None:
        rope_rotary_cos_sin = make_dummy_rope_rotary_cos_sin(
            max_seq_len, head_dim, device
        )

    kv_caches = [
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
    ]

    ctx_len = torch.full((batch_size,), trace_seq_len, device=device, dtype=torch.int32)
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (batch_size, 1),
        trace_seq_len - 1,
        device=device,
        dtype=torch.int64,
    )

    flat_tensors = (
        trace_embs,
        rope_rotary_cos_sin,
        ctx_len.contiguous(),
        kvcache_start_index,
        last_token_ids,
        *[kv.contiguous() for kv in kv_caches],
    )
    return flat_tensors, trace_seq_len


def make_language_edge_input_specs(
    input_names,
    sample_inputs,
    *,
    batch_size: int,
    max_seq_len: int,
    static_prefill_seq_len: bool = False,
):
    """Build ``torch_tensorrt.Input`` specs for Edge-LLM language engine export.

    Profile 0 is prefill (context); profile 1 is decode (generation). Shapes
    follow ``LLMBuilder::setupVanillaProfiles()`` / Edge-LLM ONNX export.

    ``inputs_embeds`` declares disjoint optimization profiles with a shared
    ``seq_len`` export symbol. ``kvcache_start_index`` uses a single dynamic
    range (min=0) on both profiles because ``Input.profiles`` rejects min=0.
    All other bindings reuse static trace shapes on every profile.
    """
    batch_size = int(batch_size)
    max_seq_len = int(max_seq_len)
    opt_prefill_seq = (
        max_seq_len
        if static_prefill_seq_len
        else language_edge_trace_seq_len(max_seq_len)
    )
    min_prefill_seq = max_seq_len if static_prefill_seq_len else 1
    hidden_size = int(sample_inputs[0].shape[2])

    prefill_profile = {
        "min": (1, min_prefill_seq, hidden_size),
        "opt": (batch_size, opt_prefill_seq, hidden_size),
        "max": (batch_size, max_seq_len, hidden_size),
    }
    decode_profile = {
        "min": (1, 1, hidden_size),
        "opt": (batch_size, 1, hidden_size),
        "max": (batch_size, 1, hidden_size),
    }

    kv_template = next(
        (tensor for name, tensor in zip(input_names, sample_inputs) if name.startswith("past_key_values_")),
        None,
    )
    kv_cache_profile = None
    if kv_template is not None:
        kv_cache_profile = {
            "min": (1, 2, int(kv_template.shape[2]), 1, int(kv_template.shape[4])),
            "opt": (
                batch_size,
                2,
                int(kv_template.shape[2]),
                max_seq_len,
                int(kv_template.shape[4]),
            ),
            "max": (
                batch_size,
                2,
                int(kv_template.shape[2]),
                max_seq_len,
                int(kv_template.shape[4]),
            ),
        }

    specs = []
    for name, tensor in zip(input_names, sample_inputs):
        if name == "inputs_embeds":
            if static_prefill_seq_len:
                specs.append(
                    torch_tensorrt.Input(
                        shape=tuple(tensor.shape),
                        dtype=tensor.dtype,
                        format=torch.contiguous_format,
                        name=name,
                    )
                )
            else:
                specs.append(
                    torch_tensorrt.Input(
                        profiles=[prefill_profile, decode_profile],
                        shared_dims={1: "seq_len"},
                        dtype=tensor.dtype,
                        format=torch.contiguous_format,
                        name=name,
                    )
                )
        elif name == "kvcache_start_index":
            specs.append(
                torch_tensorrt.Input(
                    min_shape=(0,),
                    opt_shape=(1,),
                    max_shape=(max(batch_size, 1),),
                    dtype=tensor.dtype,
                    format=torch.contiguous_format,
                    name=name,
                )
            )
        elif name.startswith("past_key_values_"):
            if kv_cache_profile is None:
                raise ValueError("past_key_values_* inputs require a KV cache template tensor")
            specs.append(
                torch_tensorrt.Input(
                    profiles=[kv_cache_profile, kv_cache_profile],
                    dtype=tensor.dtype,
                    format=torch.contiguous_format,
                    name=name,
                )
            )
        else:
            specs.append(
                torch_tensorrt.Input(
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    format=torch.contiguous_format,
                    name=name,
                )
            )
    return tuple(specs)

def language_edge_trt_settings(**overrides):
    """Compile kwargs for Edge-LLM language engines (dynamic seq + AttentionPlugin).

    Pass the returned dict as ``trt_settings`` to ``save_trt_engine_module``.
    Base settings (TF32, workspace, ``require_full_compilation``, etc.) still
    come from ``compile.py``; this adds only language-specific overrides for
    PR #4325 multi-profile + plugin attention conversion.
    """
    settings = dict(DEFAULT_LANGUAGE_TRT_SETTINGS)
    settings["use_explicit_typing"] = False
    settings.update(overrides)
    return settings


@torch.no_grad()
def save_language_engine_for_edge_llm(
    engine_dir: str | pathlib.Path,
    spec: LanguageEngineSpec,
) -> pathlib.Path:
    """Export ``language.engine`` from a populated ``LanguageEngineSpec``."""
    device = spec.prefix_embs.device
    dtype = spec.export_dtype
    prefix_embs = spec.prefix_embs.to(device=device, dtype=dtype).contiguous()

    lm_wrapper = make_plugin_lm_causal_wrapper(
        spec.decoder,
        spec.config,
        spec.lm_head,
        select_layer=spec.select_layer,
        enable_bidirectional_prefill=spec.enable_bidirectional_prefill,
        log_prefix=spec.log_prefix,
    ).to(device=device, dtype=dtype).eval()

    sample_inputs, _trace_seq_len = make_language_edge_flat_tensors(
        prefix_embs,
        batch_size=spec.batch_size,
        max_seq_len=spec.max_seq_len,
        num_layers=spec.num_layers,
        num_key_value_heads=spec.num_key_value_heads,
        head_dim=spec.head_dim,
        device=device,
        dtype=dtype,
        static_prefill_seq_len=spec.static_prefill_seq_len,
    )

    input_names = list(spec.io.input_names) + [
        f"past_key_values_{i}" for i in range(spec.num_layers)
    ]
    input_specs = make_language_edge_input_specs(
        input_names,
        sample_inputs,
        batch_size=spec.batch_size,
        max_seq_len=spec.max_seq_len,
        static_prefill_seq_len=spec.static_prefill_seq_len,
    )

    with torch.no_grad():
        example_output = lm_wrapper(*sample_inputs)

    output_names = language_edge_output_names(spec.io.output_names, spec.num_layers)

    engine_path = save_trt_engine_module(
        lm_wrapper,
        sample_inputs,
        engine_dir,
        engine_file="language.engine",
        model_type=spec.model_type,
        component="language",
        input_names=input_names,
        output_names=output_names,
        example_output=example_output,
        extra_config=language_edge_llm_config(
            spec.config,
            max_seq_len=spec.max_seq_len,
            batch_size=spec.batch_size,
            num_layers=spec.num_layers,
            context_hidden_size=spec.context_hidden_size,
            image_token_id=spec.image_token_id,
        ),
        input_specs=input_specs,
        flat_tensors=sample_inputs,
        trt_settings=spec.trt_settings,
    )
    free_cuda_memory(lm_wrapper, spec.decoder, spec.lm_head, spec.language_model)
    return engine_path


# ---------------------------------------------------------------------------
# Edge-LLM engine config (language/config.json)
# ---------------------------------------------------------------------------

def language_edge_llm_config(
    config,
    *,
    max_seq_len: int,
    batch_size: int,
    num_layers: int | None = None,
    max_lora_rank: int = 0,
    trt_native_ops: bool = False,
    context_hidden_size: int | None = None,
    image_token_id: int | None = None,
) -> dict:
    """Build language/config.json fields consumed by LLMEngineRunner at runtime."""
    config_dict = config_to_dict(config)
    head_dim = language_head_dim(config)
    num_hidden_layers = int(num_layers or config_dict["num_hidden_layers"])
    max_position_embeddings = int(
        config_dict.get("max_position_embeddings", max_seq_len)
    )

    edge_config = {
        "vocab_size": int(config_dict["vocab_size"]),
        "max_position_embeddings": max_position_embeddings,
        "hidden_size": int(config_dict["hidden_size"]),
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(config_dict["num_attention_heads"]),
        "num_key_value_heads": int(config_dict["num_key_value_heads"]),
        "head_dim": head_dim,
        # Legacy / tooling fields kept alongside Edge-LLM runner fields.
        "max_seq_len": int(max_seq_len),
        "batch_size": int(batch_size),
        "num_layers": num_hidden_layers,
        "builder_config": {
            "max_batch_size": int(batch_size),
            "max_input_len": int(max_seq_len),
            "max_kv_cache_capacity": int(max_seq_len),
            "max_lora_rank": int(max_lora_rank),
            "eagle_base": False,
            "eagle_draft": False,
            "context_emb": context_hidden_size is not None,
            "trt_native_ops": bool(trt_native_ops),
        },
    }
    if context_hidden_size is not None:
        edge_config["context_hidden_size"] = int(context_hidden_size)
    if image_token_id is not None:
        edge_config["image_token_id"] = int(image_token_id)
    edge_config.update(export_rope_fields(config_dict))

    try:
        from tensorrt_edgellm.version import __version__ as edgellm_version
    except ImportError:
        edgellm_version = None
    if edgellm_version is not None:
        edge_config["edgellm_version"] = edgellm_version

    return edge_config
