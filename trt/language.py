import logging

from typing import Any

import torch
import torch.nn as nn

from trt.attention import PluginAttention
from trt.utils import build_prefix_inputs
from trt.compile import compile_trt_module
from trt.plugin_utils import set_plugin_config_from_model

FP16 = torch.float16

logger = logging.getLogger(__name__)

def _as_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def make_dummy_rope_rotary_cos_sin(
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Placeholder RoPE cache for export/compile tracing (values are ignored)."""
    return torch.randn(
        1,
        int(max_seq_len),
        int(head_dim),
        dtype=torch.float32,
        device=device,
    )


def make_prefill_kvcache_start_index(device: torch.device) -> torch.Tensor:
    """Empty tensor signals fresh prefill with no existing KV cache."""
    return torch.empty(0, dtype=torch.int32, device=device)


def rotary_dim_from_config(config) -> int:
    head_dim = language_head_dim(config)
    partial = float(getattr(config, "partial_rotary_factor", 1.0) or 1.0)
    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim > head_dim:
        rotary_dim = head_dim
    return rotary_dim


def _resolve_rotary_emb(language_model: nn.Module) -> nn.Module | None:
    for module in (language_model, getattr(language_model, "model", None)):
        if module is None:
            continue
        rotary_emb = getattr(module, "rotary_emb", None)
        if rotary_emb is not None:
            return rotary_emb
    return None


def _plugin_rope_layout(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    max_seq_len: int,
    rotary_dim: int,
) -> torch.Tensor:
    """Pack cos/sin into AttentionPlugin layout: [:, :, :half]=cos, [:, :, half:]=sin."""
    half = rotary_dim // 2
    cos_half = cos[..., :half].float()
    sin_half = sin[..., :half].float()
    cache = torch.cat(
        [cos_half[:, :max_seq_len], sin_half[:, :max_seq_len]],
        dim=-1,
    )
    if cache.shape[0] != 1:
        cache = cache[:1]
    return cache.contiguous()


def make_normal_rope_rotary_cos_sin(
    max_seq_len: int,
    rotary_dim: int,
    *,
    rope_theta: float,
    rotary_scale: float = 1.0,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE cache using the same formula as Edge-LLM initializeNormalRopeCosSin."""
    half = rotary_dim // 2
    zid = torch.arange(half, device=device, dtype=torch.float32)
    inv_denominator = float(rope_theta) ** (2 * zid / float(rotary_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    angles = positions * float(rotary_scale) / inv_denominator.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    return torch.cat([cos, sin], dim=-1).unsqueeze(0).to(dtype=torch.float32)


def make_rope_rotary_cos_sin_from_config(
    config,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE cache from HF config fields (matches LLMEngineRunner default/dynamic RoPE)."""
    config_dict = _config_to_dict(config)
    rope_params = _select_rope_parameters(config_dict)
    rope_type = None
    if rope_params:
        rope_type = rope_params.get("rope_type", rope_params.get("type"))

    if rope_type == "longrope":
        raise NotImplementedError(
            "longrope config cache is not implemented; pass language_model= to "
            "make_rope_rotary_cos_sin() instead"
        )
    if rope_type not in (None, "default", "dynamic", "llama3"):
        raise NotImplementedError(
            f"RoPE type {rope_type!r} requires language_model= and position_ids="
        )

    rotary_dim = rotary_dim_from_config(config)
    rope_theta = _export_rope_fields(config_dict)["rope_theta"]
    return make_normal_rope_rotary_cos_sin(
        max_seq_len,
        rotary_dim,
        rope_theta=float(rope_theta),
        rotary_scale=1.0,
        device=device,
    )


@torch.no_grad()
def make_rope_rotary_cos_sin_from_model(
    language_model: nn.Module,
    config,
    max_seq_len: int,
    device: torch.device,
    *,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build plugin RoPE cache from the model's rotary_emb (best eager parity)."""
    rotary_emb = _resolve_rotary_emb(language_model)
    if rotary_emb is None:
        raise AttributeError("language_model has no rotary_emb")

    rotary_dim = rotary_dim_from_config(config)
    if position_ids is None:
        position_ids = torch.arange(max_seq_len, device=device, dtype=torch.long).unsqueeze(0)
    else:
        position_ids = position_ids.to(device=device)

    seq_len = int(position_ids.shape[-1])
    dummy = torch.ones(
        int(position_ids.shape[0] if position_ids.ndim >= 2 else 1),
        seq_len,
        1,
        device=device,
        dtype=torch.float16,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    return _plugin_rope_layout(
        cos,
        sin,
        max_seq_len=max_seq_len,
        rotary_dim=rotary_dim,
    )


@torch.no_grad()
def make_rope_rotary_cos_sin(
    config,
    max_seq_len: int,
    device: torch.device,
    *,
    language_model: nn.Module | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build real RoPE cache for inference/parity (not export tracing).

    Prefers the model's ``rotary_emb`` when available for eager parity. Falls back
    to config-based generation that matches Edge-LLM ``initializeNormalRopeCosSin``.
    """
    if language_model is not None:
        try:
            return make_rope_rotary_cos_sin_from_model(
                language_model,
                config,
                max_seq_len,
                device,
                position_ids=position_ids,
            )
        except (AttributeError, NotImplementedError, RuntimeError) as exc:
            logger.warning(
                "Building RoPE from model rotary_emb failed (%s); using config kernel",
                exc,
            )
    return make_rope_rotary_cos_sin_from_config(config, max_seq_len, device)


class FlatKVLanguageEngineWrapper(nn.Module):
    """Expose list-style KV caches as flat TensorRT engine inputs."""

    def __init__(self, plugin_language: nn.Module):
        super().__init__()
        self.plugin_language = plugin_language

    def forward(self, inputs_embeds, rope_rotary_cos_sin, ctx_len, kvcache_start_index, *kv_caches):
        return self.plugin_language(
            inputs_embeds,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            list(kv_caches),
        )

class PluginLMHiddenWrapper(nn.Module):
    """Shared plugin-attention LM prefill.

    Returns hidden states, and optionally prefix K/V.
    """

    def __init__(
        self,
        lm: nn.Module,
        *,
        num_ds: int = 0
    ):
        super().__init__()
        self.lm = lm
        self.num_ds = int(num_ds)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        ctx_len: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_caches: list[torch.Tensor],
        ds_stack: torch.Tensor | None = None,
    ):
        hidden = _as_tensor(inputs_embeds)
        seq_len = inputs_embeds.shape[1]
        new_kvs = []

        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                past_key_value=kv_caches[i],
                ctx_len=ctx_len,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = _as_tensor(hidden)
            hidden = residual + hidden

            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = _as_tensor(layer.mlp(hidden))
            hidden = residual + hidden

            new_kvs.append(kv)

            if self.num_ds > 0 and ds_stack is not None and i < self.num_ds:
                hidden = hidden + ds_stack[i, :, :seq_len, :]

        hidden = _as_tensor(self.lm.norm(hidden))

        prefix_k = torch.stack(
            [kv[:, 0, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        prefix_v = torch.stack(
            [kv[:, 1, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )

        return hidden, prefix_k, prefix_v

# TODO: 
class GROOTContextProjectionWrapper(nn.Module):
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
    """Single exported engine: inputs_embeds -> LM -> GR00T context_embs."""

    def __init__(
        self,
        lm_wrapper: PluginLMHiddenWrapper,
        context_projection: GROOTContextProjectionWrapper,
    ):
        super().__init__()
        self.lm_wrapper = lm_wrapper
        self.context_projection = context_projection

    def forward(self, inputs_embeds, rope_rotary_cos_sin, ctx_len, kvcache_start_index, kv_caches):
        hidden, prefix_k, prefix_v = self.lm_wrapper(
            inputs_embeds,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            kv_caches,
        )
        return self.context_projection(hidden)

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
):
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

    rope_rotary_cos_sin = make_dummy_rope_rotary_cos_sin(
        max_seq_len, head_dim, device
    )

    ctx_len = torch.full(
        (batch_size,),
        ctx_seq_len,
        device=device,
        dtype=torch.int32,
    )
    kvcache_start_index = make_prefill_kvcache_start_index(device)

    trt_language = compile_trt_module(
        plugin_language,
        (
            inputs_embeds.to(device=device, dtype=dtype),
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            kv_caches,
        ),
        settings,
    )

    return trt_language, max_seq_len

@torch.no_grad()
def run_prefix_language_eager(language_model, prefix_embs, attention_mask, position_ids):
    lm_dtype = next(language_model.parameters()).dtype
    prefix_embs = prefix_embs.to(dtype=lm_dtype)
    out = language_model(
        inputs_embeds=prefix_embs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
    )
    cache = out.past_key_values
    prefix_k = torch.stack([layer.keys for layer in cache.layers], dim=0)
    prefix_v = torch.stack([layer.values for layer in cache.layers], dim=0)
    return out.last_hidden_state, prefix_k, prefix_v

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


def _select_valid_token_rows(hidden, prefix_pad_masks=None, max_logit_tokens=16):
    if prefix_pad_masks is None:
        rows = hidden.reshape(-1, hidden.shape[-1])
        desc = f"{rows.shape[0]} total token rows"
    else:
        valid = prefix_pad_masks.to(device=hidden.device, dtype=torch.bool)
        rows = torch.cat(
            [hidden[b, valid[b], :] for b in range(valid.shape[0])],
            dim=0,
        )
        desc = f"{rows.shape[0]} valid token rows"

    if max_logit_tokens is not None and rows.shape[0] > max_logit_tokens:
        rows = rows[-max_logit_tokens:]
        desc = f"{desc}; comparing last {max_logit_tokens}"

    return rows, desc

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
    kvcache_start_index = make_prefill_kvcache_start_index(device)
    trt_hidden, trt_k, trt_v = trt_language(
        trt_prefix_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        kv_caches,
    )

    _smoke_tensor_health("LM plugin smoke-check eager hidden", eager_hidden)
    _smoke_tensor_health("LM plugin smoke-check TRT hidden", trt_hidden)

    eager_rows, desc = _select_valid_token_rows(eager_hidden, prefix_pad_masks, max_logit_tokens)
    trt_rows, _ = _select_valid_token_rows(trt_hidden, prefix_pad_masks, max_logit_tokens)

    eager_logits = lm_head(eager_rows.to(device=device, dtype=head_dtype))
    trt_logits = lm_head(trt_rows.to(device=device, dtype=head_dtype))

    print("LM plugin smoke-check logits rows:", desc)
    print("LM plugin smoke-check logits shape:", tuple(trt_logits.shape))
    _smoke_error_metrics("LM plugin smoke-check logits", trt_logits, eager_logits, include_top1=True)

    cache = eager_out.past_key_values
    eager_k = torch.stack([layer.keys for layer in cache.layers], dim=0)
    eager_v = torch.stack([layer.values for layer in cache.layers], dim=0)

    _smoke_error_metrics("LM plugin smoke-check prefix_k", trt_k, eager_k)
    _smoke_error_metrics("LM plugin smoke-check prefix_v", trt_v, eager_v)

def _install_plugin_attention(
    lm: nn.Module,
    config,
    enable_bidirectional_prefill: int = 1,
) -> None:
    for i, layer in enumerate(lm.layers):
        layer.self_attn = PluginAttention(
            layer.self_attn,
            config,
            layer_idx=i,
            enable_bidirectional_prefill=enable_bidirectional_prefill,
        ).eval()

def make_plugin_lm_hidden_wrapper(
    decoder: nn.Module,
    config,
    *,
    max_seq_len: int,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
    enable_bidirectional_prefill: int = 1,
    log_prefix: str = "",
) -> PluginLMHiddenWrapper:
    del position_ids, max_seq_len, device

    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )

    prefix = f"{log_prefix} " if log_prefix else ""
    print(f"{prefix}head_dim:", head_dim)

    _install_plugin_attention(
        decoder,
        config,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
    )

    return PluginLMHiddenWrapper(
        decoder,
        num_ds=0
    ).eval()


def language_head_dim(config) -> int:
    return int(getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    ))


def _config_to_dict(config) -> dict:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError(f"Expected a HuggingFace-style config with to_dict(), got {type(config)!r}")


def _select_rope_parameters(config_dict: dict) -> dict:
    rope_parameters = config_dict.get("rope_parameters")
    if rope_parameters is None:
        rope_parameters = config_dict.get("rope_scaling")
    if not isinstance(rope_parameters, dict):
        return {}

    layer_types = config_dict.get("layer_types")
    if layer_types and set(rope_parameters.keys()).issubset(set(layer_types)):
        full_attention = rope_parameters.get("full_attention")
        if isinstance(full_attention, dict):
            return dict(full_attention)
        for layer_type in layer_types:
            layer_params = rope_parameters.get(layer_type)
            if isinstance(layer_params, dict):
                return dict(layer_params)
        return {}

    full_attention = rope_parameters.get("full_attention")
    if isinstance(full_attention, dict):
        return dict(full_attention)

    return dict(rope_parameters)


def _normalize_rope_scaling(rope_params: dict) -> dict:
    rope_scaling = dict(rope_params)
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type is not None:
        rope_scaling.setdefault("rope_type", rope_type)
        rope_scaling.setdefault("type", rope_type)
    return rope_scaling


def _export_rope_fields(config_dict: dict) -> dict:
    """RoPE fields for LLMEngineRunner::collectRopeConfig / initializeRopeCosSinCache."""
    rope_fields: dict = {}
    rope_params = _select_rope_parameters(config_dict)

    if "rope_theta" in config_dict:
        rope_fields["rope_theta"] = config_dict["rope_theta"]
    elif "rope_theta" in rope_params:
        rope_fields["rope_theta"] = rope_params["rope_theta"]
    else:
        raise KeyError("rope_theta not found in language model config")

    if rope_params:
        rope_fields["rope_scaling"] = _normalize_rope_scaling(rope_params)
    else:
        rope_fields["rope_scaling"] = None

    if rope_fields["rope_scaling"] is not None:
        rope_type = rope_fields["rope_scaling"].get("rope_type")
        if rope_type == "longrope":
            original_max = rope_fields["rope_scaling"].get(
                "original_max_position_embeddings",
                config_dict.get("original_max_position_embeddings"),
            )
            if original_max is None:
                raise KeyError(
                    "original_max_position_embeddings required for longrope scaling"
                )
            rope_fields["original_max_position_embeddings"] = original_max

    if "partial_rotary_factor" in config_dict:
        rope_fields["partial_rotary_factor"] = config_dict["partial_rotary_factor"]
    elif "partial_rotary_factor" in rope_params:
        rope_fields["partial_rotary_factor"] = rope_params["partial_rotary_factor"]
    else:
        rope_fields["partial_rotary_factor"] = 1.0

    return rope_fields


def language_edge_llm_config(
    config,
    *,
    max_seq_len: int,
    batch_size: int,
    num_layers: int | None = None,
    max_lora_rank: int = 0,
    trt_native_ops: bool = False,
) -> dict:
    """Build language/config.json fields consumed by LLMEngineRunner at runtime."""
    config_dict = _config_to_dict(config)
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
            "trt_native_ops": bool(trt_native_ops),
        },
    }
    edge_config.update(_export_rope_fields(config_dict))

    try:
        from tensorrt_edgellm.version import __version__ as edgellm_version
    except ImportError:
        edgellm_version = None
    if edgellm_version is not None:
        edge_config["edgellm_version"] = edgellm_version

    return edge_config