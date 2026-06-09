import copy

import torch
import torch.nn as nn

from trt.attention import PluginAttention
from trt.utils import build_prefix_inputs
from trt.compile import compile_trt_module

FP16 = torch.float16

def _as_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x

class PluginPrefixLMWrapper(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        *,
        lm_head: nn.Module | None = None,
        num_ds: int = 0,
        return_logits: bool = False,
        return_hidden: bool = False,
        return_prefix_kv: bool = True,
    ):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.num_ds = int(num_ds)
        self.return_logits = bool(return_logits)
        self.return_hidden = bool(return_hidden)
        self.return_prefix_kv = bool(return_prefix_kv)

        if self.return_logits and self.lm_head is None:
            raise ValueError("lm_head is required when return_logits=True")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: list[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor | None = None,
    ):
        hidden = _as_tensor(inputs_embeds)
        seq_len = inputs_embeds.shape[1]
        new_kvs: list[torch.Tensor] = []

        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                past_key_value=kv_caches[i],
                ctx_len=ctx_len,
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

        outputs = []

        if self.return_hidden:
            outputs.append(hidden)

        if self.return_logits:
            logits = self.lm_head(hidden)
            outputs.append(logits)

        if self.return_prefix_kv:
            prefix_k = torch.stack(
                [kv[:, 0, :, :seq_len, :] for kv in new_kvs],
                dim=0,
            )
            prefix_v = torch.stack(
                [kv[:, 1, :, :seq_len, :] for kv in new_kvs],
                dim=0,
            )
            outputs.extend([prefix_k, prefix_v])

        if outputs:
            return tuple(outputs)

        return hidden, new_kvs


class GROOTPluginContextWrapper(nn.Module):
    def __init__(self, decoder, eagle_linear, vlln, vl_self_attention):
        super().__init__()
        self.decoder = decoder
        self.eagle_linear = eagle_linear
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, inputs_embeds, kv_caches, ctx_len):
        hidden = _as_tensor(inputs_embeds)
        seq_len = inputs_embeds.shape[1]

        for i, layer in enumerate(self.decoder.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, _ = layer.self_attn(
                hidden_states=hidden,
                past_key_value=kv_caches[i],
                ctx_len=ctx_len,
            )
            hidden = _as_tensor(hidden)
            hidden = residual + hidden

            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = _as_tensor(layer.mlp(hidden))
            hidden = residual + hidden

        hidden = _as_tensor(self.decoder.norm(hidden))

        context_embs = self.eagle_linear(hidden)

        vlln_weight = getattr(self.vlln, "weight", None)
        if vlln_weight is not None:
            context_embs = context_embs.to(dtype=vlln_weight.dtype)

        context_embs = self.vlln(context_embs)
        context_embs = self.vl_self_attention(context_embs)
        return context_embs

@torch.no_grad()
def run_vlm_preprocessing(
    core,
    images,
    img_masks,
    tokens,
    masks,
    trt_vision=None,
    *,
    dtype=torch.float16,
):
    image_embs = [
        trt_vision(image) if trt_vision is not None else core.paligemma_with_expert.embed_image(image)
        for image in images
    ]

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_prefix_inputs(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
    )

    return (
        prefix_embs.to(dtype),
        prefix_pad_masks,
        prefix_attention_mask,
        prefix_position_ids,
    )

def compile_groot_lm_trt_with_plugin(
    core,
    input_embs,
    *,
    device,
    position_ids=None,
    settings,
):
    max_seq_len = input_embs.shape[1]

    plugin_language = make_groot_plugin_language(
        core,
        max_seq_len=max_seq_len,
        device=device,
        position_ids=position_ids,
        attention_cls=PluginAttention,
    )

    kv_caches = make_groot_language_kv_caches(
        core,
        batch_size=input_embs.shape[0],
        max_seq_len=max_seq_len,
        device=device,
    )

    ctx_len = torch.full(
        (input_embs.shape[0],),
        max_seq_len,
        device=device,
        dtype=torch.int32,
    )

    trt_language = compile_trt_module(
        plugin_language,
        (input_embs.to(torch.float16), kv_caches, ctx_len),
        settings,
    )

    return trt_language, max_seq_len

def compile_lm_trt_with_plugin(
    core,
    prefix_embs,
    *,
    device,
    position_ids=None,
    settings
):
    max_seq_len = prefix_embs.shape[1]

    plugin_language = make_pi05_plugin_language(
        core,
        max_seq_len=max_seq_len,
        device=device,
        position_ids=position_ids,
        attention_cls=PluginAttention,
    )

    kv_caches = make_pi05_language_kv_caches(
        core,
        batch_size=prefix_embs.shape[0],
        max_seq_len=max_seq_len,
        device=device,
    )

    ctx_len = torch.full(
        (prefix_embs.shape[0],),
        max_seq_len,
        device=device,
        dtype=torch.int32,
    )

    trt_language = compile_trt_module(
        plugin_language,
        (prefix_embs, kv_caches, ctx_len),
        settings,
    )

    return trt_language, max_seq_len

@torch.no_grad()
def run_prefix_language_eager(language_model, prefix_embs, attention_mask, position_ids):
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

@torch.no_grad()
def run_prefix_plugin_language(
    trt_language,
    core,
    prefix_embs,
    *,
    max_seq_len,
    device,
    attention_mask=None,
    prefix_pad_masks=None,
    return_hidden=False,
):
    prefix_embs = prefix_embs.to(torch.float16)

    kv_caches = make_pi05_language_kv_caches(
        core,
        batch_size=prefix_embs.shape[0],
        max_seq_len=max_seq_len,
        device=device,
    )

    ctx_len = torch.full(
        (prefix_embs.shape[0],),
        prefix_embs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    if prefix_pad_masks is not None or attention_mask is not None:
        print("[plugin prefix] sanity check: using full seq_len ctx_len", ctx_len.detach().cpu().tolist())

    out = trt_language(prefix_embs, kv_caches, ctx_len)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        hidden, prefix_k, prefix_v = out
        if return_hidden:
            return hidden, prefix_k, prefix_v
        return prefix_k, prefix_v
    return out


@torch.no_grad()
def run_groot_plugin_language(
    trt_language,
    core,
    input_embs,
    *,
    max_seq_len,
    device,
):
    input_embs = input_embs.to(device=device, dtype=torch.float16)

    kv_caches = make_groot_language_kv_caches(
        core,
        batch_size=input_embs.shape[0],
        max_seq_len=max_seq_len,
        device=device,
    )

    ctx_len = torch.full(
        (input_embs.shape[0],),
        input_embs.shape[1],
        device=device,
        dtype=torch.int32,
    )

    return trt_language(input_embs, kv_caches, ctx_len)

def _groot_decoder(language_model):
    # Eagle uses HF CausalLM wrappers like Qwen2ForCausalLM/Qwen3ForCausalLM.
    return getattr(language_model, "model", language_model)


def make_groot_language_kv_caches(core, batch_size, max_seq_len, device):
    language_model = core.backbone.eagle_model.language_model
    decoder = _groot_decoder(language_model)
    cfg = language_model.config

    head_dim = getattr(
        cfg,
        "head_dim",
        cfg.hidden_size // cfg.num_attention_heads,
    )

    return [
        torch.zeros(
            batch_size,
            2,
            cfg.num_key_value_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        for _ in range(len(decoder.layers))
    ]

def make_groot_plugin_language(
    core,
    max_seq_len,
    device,
    position_ids=None,
    attention_cls=PluginAttention,
):
    eagle = core.backbone.eagle_model

    language_model = copy.deepcopy(eagle.language_model).to(
        device=device,
        dtype=torch.float16,
    ).eval()

    decoder = _groot_decoder(language_model)

    if position_ids is None:
        position_ids = torch.arange(max_seq_len, device=device).view(1, max_seq_len)

    position_ids = position_ids.to(device=device)[:, :max_seq_len]

    cfg = language_model.config
    head_dim = getattr(
        cfg,
        "head_dim",
        cfg.hidden_size // cfg.num_attention_heads,
    )

    with torch.no_grad():
        dummy = torch.ones(
            position_ids.shape[0],
            max_seq_len,
            cfg.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        cos, sin = decoder.rotary_emb(dummy, position_ids)

        h2 = head_dim // 2
        rope_cache = torch.cat(
            [
                cos[:, :max_seq_len, :h2].float(),
                sin[:, :max_seq_len, :h2].float(),
            ],
            dim=-1,
        )

    print("groot rope_cache shape/dtype:", rope_cache.shape, rope_cache.dtype)
    print("groot head_dim:", head_dim)

    _install_plugin_attention(
        decoder,
        cfg,
        rope_cache,
        attention_cls=attention_cls,
        enable_bidirectional_prefill=0,
    )

    return GROOTPluginContextWrapper(
        decoder,
        copy.deepcopy(core.backbone.eagle_linear).to(device=device, dtype=torch.float16).eval(),
        copy.deepcopy(core.action_head.vlln).to(device=device, dtype=torch.float16).eval(),
        copy.deepcopy(core.action_head.vl_self_attention).to(device=device, dtype=torch.float16).eval(),
    ).eval()

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

    trt_hidden, trt_k, trt_v = run_prefix_plugin_language(
        trt_language,
        core,
        prefix_embs,
        max_seq_len=max_seq_len,
        device=device,
        attention_mask=attention_mask,
        prefix_pad_masks=prefix_pad_masks,
        return_hidden=True,
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


def _build_rope_cache(
    lm: nn.Module,
    S_input: int,
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-compute concatenated ``(cos, sin)`` RoPE cache for all positions up to ``max_seq_len``."""
    with torch.no_grad():
        d_eff = torch.arange(S_input, max_seq_len, device=device).float()
        d_eff = d_eff + rope_deltas.to(device).float().squeeze()
        d_3d = d_eff.view(1, 1, -1).expand(3, 1, -1).long()
        full_pos = torch.cat([position_ids.to(device), d_3d], dim=2)
        cos, sin = lm.rotary_emb(torch.ones(1, device=device, dtype=FP16), full_pos)
        h2 = head_dim // 2
        rope_cache = torch.cat(
            [cos[:, :max_seq_len, :h2].float(), sin[:, :max_seq_len, :h2].float()],
            dim=-1,
        )
    return rope_cache

def _install_plugin_attention(
    lm: nn.Module,
    config,
    rope_cache: torch.Tensor,
    attention_cls=PluginAttention,
    enable_bidirectional_prefill: int = 1,
) -> None:
    for i, layer in enumerate(lm.layers):
        layer.self_attn = attention_cls(
            layer.self_attn,
            config,
            layer_idx=i,
            rope_cache=rope_cache,
            enable_bidirectional_prefill=enable_bidirectional_prefill,
        ).eval()

def make_pi05_plugin_language(
    core: nn.Module,
    max_seq_len: int,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
    attention_cls=PluginAttention,
) -> PluginPrefixLMWrapper:
    lm = copy.deepcopy(
        core.paligemma_with_expert.paligemma.model.language_model
    ).to(device=device, dtype=torch.float16).eval()

    if position_ids is None:
        position_ids = torch.arange(max_seq_len, device=device).view(1, max_seq_len)

    position_ids = position_ids.to(device=device)[:, :max_seq_len]

    with torch.no_grad():
        dummy = torch.ones(
            position_ids.shape[0],
            max_seq_len,
            lm.config.hidden_size,
            device=device,
            dtype=FP16,
        )
        cos, sin = lm.rotary_emb(dummy, position_ids)

        h2 = lm.config.head_dim // 2
        rope_cache = torch.cat(
            [cos[:, :max_seq_len, :h2].float(), sin[:, :max_seq_len, :h2].float()],
            dim=-1,
        )

    print("rope_cache shape/dtype:", rope_cache.shape, rope_cache.dtype)
    print("head_dim:", lm.config.head_dim)

    _install_plugin_attention(lm, lm.config, rope_cache, attention_cls=attention_cls)

    return PluginPrefixLMWrapper(
        lm,
        num_ds=0,
        return_logits=False,
        return_hidden=True,
        return_prefix_kv=True,
    ).eval()

def make_pi05_language_kv_caches(core: nn.Module, batch_size: int, max_seq_len: int, device: torch.device):
    cfg = core.paligemma_with_expert.paligemma.model.language_model.config
    return [
        torch.zeros(
            batch_size,
            2,
            cfg.num_key_value_heads,
            max_seq_len,
            cfg.head_dim,
            device=device,
            dtype=torch.float16,
        )
        for _ in range(cfg.num_hidden_layers)
    ]