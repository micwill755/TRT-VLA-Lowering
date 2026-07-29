"""Cosmos3-Edge policy export modules for ``Cosmos3Runtime``.

Torch-TRT counterparts of MR ``modeling_und_prefill`` / ``modeling_gen``:
UND is prefilled once (frozen per-layer K/V); GEN is one denoise step that
cross-attends to those K/V. Uses real SDPA + RoPE (MR ONNX ops are TRT stubs).
"""

from __future__ import annotations

import math
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    gen_config_from_transformer,
    und_config_from_transformer,
)
from weights import (
    load_gen_weights,
    load_und_weights,
    split_transformer_weights,
)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def apply_rope_packed(
    x: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE from packed ``[B, S, head_dim] = concat(cos, sin)`` (half/half).

    ``x`` is ``[B, H, S, D]``. ``position_ids`` is ``[B, S]`` (used when the rope
    cache is longer than the active sequence; for dense 0..S-1 it is a no-op gather).
    """
    bsz, _, seq, head_dim = x.shape
    half = head_dim // 2
    # Gather rope rows by position id so dynamic und_len / gen_len stay correct.
    flat_pos = position_ids.reshape(bsz, seq, 1).expand(bsz, seq, head_dim)
    rope = torch.gather(rope_rotary_cos_sin, 1, flat_pos.to(torch.int64))
    cos = torch.cat([rope[..., :half], rope[..., :half]], dim=-1).unsqueeze(1).to(dtype=x.dtype)
    sin = torch.cat([rope[..., half:], rope[..., half:]], dim=-1).unsqueeze(1).to(dtype=x.dtype)
    return x * cos + _rotate_half(x) * sin


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        return (self.weight.float() * x_f).to(dtype)


class _UndAttn(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv: int,
        head_dim: int,
        *,
        use_und_k_norm: bool,
        rms_eps: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.k_norm_und_for_gen = _RMSNorm(head_dim, rms_eps) if use_und_k_norm else None

    def forward(
        self,
        h: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq, _ = h.shape
        q = self.q_proj(h).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq, self.n_kv, self.head_dim).transpose(1, 2)

        # UND self-attn: raw K (no qk-norm). GEN context K: optional RMSNorm-before-RoPE.
        q_rope = apply_rope_packed(q, rope_rotary_cos_sin, position_ids)
        k_self = apply_rope_packed(k, rope_rotary_cos_sin, position_ids)
        attn = F.scaled_dot_product_attention(q_rope, k_self, v, is_causal=True, enable_gqa=True)
        out = self.o_proj(attn.transpose(1, 2).reshape(bsz, seq, -1))

        if self.k_norm_und_for_gen is not None:
            k_gen = apply_rope_packed(self.k_norm_und_for_gen(k), rope_rotary_cos_sin, position_ids)
        else:
            k_gen = k_self
        # Seq-major [B, S, H_kv, D] for Cosmos3Runtime / GEN bindings.
        return out, k_gen.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()


class _UndMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, hidden_act: str) -> None:
        super().__init__()
        if hidden_act == "relu2":
            self.act_fn = lambda x: F.relu(x).square()
        elif hidden_act == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported UND hidden_act: {hidden_act!r}")
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class _UndLayer(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.input_layernorm = _RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = _RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = _UndAttn(
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["head_dim"],
            use_und_k_norm=bool(cfg.get("use_und_k_norm_for_gen", False)),
            rms_eps=cfg["rms_norm_eps"],
        )
        self.mlp = _UndMLP(cfg["hidden_size"], cfg["intermediate_size"], cfg["hidden_act"])

    def forward(self, x, rope_rotary_cos_sin, position_ids):
        attn_out, k, v = self.self_attn(self.input_layernorm(x), rope_rotary_cos_sin, position_ids)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class Cosmos3UndPrefillExportModule(nn.Module):
    """UND prefill → per-layer ``und_k/v`` (+ ``hidden_states``) for ``Cosmos3Runtime``."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = int(cfg["num_hidden_layers"])
        self.layers = nn.ModuleList([_UndLayer(cfg) for _ in range(self.n)])
        self.norm = _RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x = inputs_embeds
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for layer in self.layers:
            x, k, v = layer(x, rope_rotary_cos_sin, attention_pos_id)
            ks.append(k)
            vs.append(v)
        return tuple(ks) + tuple(vs) + (self.norm(x),)


class _GenAttn(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.to_q = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.norm_q = _RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.norm_k = _RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> torch.Tensor:
        bsz, s_gen, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(bsz, s_gen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(hidden_states).view(bsz, s_gen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(hidden_states).view(bsz, s_gen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = apply_rope_packed(q, rope_rotary_cos_sin, attention_pos_id)
        k = apply_rope_packed(k, rope_rotary_cos_sin, attention_pos_id)
        # und_k/v arrive seq-major [B, S_und, H_kv, D]
        k_all = torch.cat([k_und.transpose(1, 2).to(k.dtype), k], dim=2)
        v_all = torch.cat([v_und.transpose(1, 2).to(v.dtype), v], dim=2)
        attn = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False, enable_gqa=True)
        return self.to_out(attn.transpose(1, 2).reshape(bsz, s_gen, -1))


class _GenMLP(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        if cfg.hidden_act == "relu2":
            self.act_fn = lambda x: F.relu(x).square()
        elif cfg.hidden_act == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported GEN hidden_act: {cfg.hidden_act!r}")
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class _GenLayer(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cross_attention = _GenAttn(cfg)
        self.input_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = _GenMLP(cfg)

    def forward(self, hidden, k_und, v_und, rope, pos):
        residual = hidden
        hidden = residual + self.cross_attention(
            self.input_layernorm(hidden), k_und, v_und, rope, pos
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _TimestepEmbedder(nn.Module):
    """Match MR ``TimestepEmbedder`` param names for weight load."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.frequency_embedding_size = cfg.frequency_embedding_size
        self.linear_1 = nn.Linear(cfg.frequency_embedding_size, cfg.hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        half = cfg.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(cfg.timestep_max_period)
            * torch.arange(0, half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.linear_2(self.act(self.linear_1(t_freq.type_as(self.linear_1.weight))))


class Cosmos3GenStepExportModule(nn.Module):
    """One GEN denoise step: video∥action tokens + frozen UND K/V → velocity preds."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.patch_latent_dim, cfg.hidden_size, bias=True)
        self.proj_out = nn.Linear(cfg.hidden_size, cfg.patch_latent_dim, bias=True)
        self.time_embedder = _TimestepEmbedder(cfg)
        self.action_modality_embed = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.action_in_weight = nn.Parameter(torch.zeros(cfg.max_action_dim, cfg.hidden_size))
        self.action_in_bias = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.action_out_weight = nn.Parameter(torch.zeros(cfg.hidden_size, cfg.max_action_dim))
        self.action_out_bias = nn.Parameter(torch.zeros(cfg.max_action_dim))
        self.layers = nn.ModuleList([_GenLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm_moe_gen = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = latents.shape
        p = self.cfg.latent_patch_size
        hp, wp = h // p, w // p
        x = latents.reshape(b, c, t, hp, p, wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)
        return x.reshape(b, t * hp * wp, p * p * c)

    def _unpatchify(self, tokens: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        b = tokens.shape[0]
        p, c = self.cfg.latent_patch_size, self.cfg.latent_channel
        hp, wp = h // p, w // p
        x = tokens.reshape(b, t, hp, wp, p, p, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        return x.reshape(b, c, t, hp * p, wp * p)

    def _forward_body(
        self,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        timestep: torch.Tensor,
        token_noisy_mask: torch.Tensor,
        action_noisy_mask: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
        und_kv: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n = cfg.num_hidden_layers
        k_und = und_kv[:n]
        v_und = und_kv[n:]
        io_type = self.proj_in.weight.dtype

        video_latent = video_latent.to(io_type)
        action_latent = action_latent.to(io_type)
        _, _, t_lat, h_lat, w_lat = video_latent.shape

        video_tokens = self.proj_in(self._patchify(video_latent))
        action_tokens = torch.matmul(action_latent, self.action_in_weight) + self.action_in_bias
        action_tokens = action_tokens + self.action_modality_embed

        t_embed = self.time_embedder(timestep.float() * cfg.timestep_scale).to(io_type)
        video_tokens = video_tokens + t_embed[:, None, :] * token_noisy_mask.to(io_type)
        action_tokens = action_tokens + t_embed[:, None, :] * action_noisy_mask.to(io_type)
        hidden = torch.cat([video_tokens, action_tokens], dim=1)

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, k_und[i], v_und[i], rope_rotary_cos_sin, attention_pos_id)

        hidden = self.norm_moe_gen(hidden)
        s_video = video_tokens.shape[1]
        video_pred = self._unpatchify(self.proj_out(hidden[:, :s_video, :]), t_lat, h_lat, w_lat).float()
        action_pred = (
            torch.matmul(hidden[:, s_video:, :], self.action_out_weight) + self.action_out_bias
        ).float()
        return video_pred, action_pred

    def forward(
        self,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        timestep: torch.Tensor,
        token_noisy_mask: torch.Tensor,
        action_noisy_mask: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
        *und_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Eager path keeps *und_kv; export wraps with named bindings via
        # ``bind_named_gen_forward`` so Cosmos3Runtime sees und_k/v_layerXX.
        return self._forward_body(
            video_latent,
            action_latent,
            timestep,
            token_noisy_mask,
            action_noisy_mask,
            rope_rotary_cos_sin,
            attention_pos_id,
            und_kv,
        )


def bind_named_gen_forward(module: Cosmos3GenStepExportModule) -> Cosmos3GenStepExportModule:
    """Replace ``forward`` with explicitly named ``und_k_layerXX`` / ``und_v_layerXX`` args.

    Torch-TRT names engine I/O from the FX placeholder names. A ``*und_kv`` signature
    becomes ``und_kv_0..N``, which breaks Cosmos3Runtime (expects ``und_k_layer00`` etc.).
    """
    n = int(module.cfg.num_hidden_layers)
    k_names = [f"und_k_layer{i:02d}" for i in range(n)]
    v_names = [f"und_v_layer{i:02d}" for i in range(n)]
    kv_params = ", ".join(k_names + v_names)
    kv_tuple = ", ".join(k_names + v_names)
    src = f"""
def forward(self, video_latent, action_latent, timestep, token_noisy_mask,
            action_noisy_mask, rope_rotary_cos_sin, attention_pos_id, {kv_params}):
    return self._forward_body(
        video_latent, action_latent, timestep, token_noisy_mask, action_noisy_mask,
        rope_rotary_cos_sin, attention_pos_id, ({kv_tuple},))
"""
    ns: dict[str, Any] = {}
    exec(src, ns)  # noqa: S102 — generated signature must match Cosmos3Runtime bindings
    module.forward = ns["forward"].__get__(module, type(module))  # type: ignore[method-assign]
    return module


class Cosmos3VaeEncoderExportModule(nn.Module):
    """``pixel_values`` → ``cond_latent`` (Cosmos3Runtime VAE binding names)."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        from trt.modules.cosmos.packing import encode_cosmos_video

        return encode_cosmos_video(self.vae, pixel_values)


def load_policy_und_prefill(
    checkpoint: str,
    dtype: torch.dtype,
) -> Cosmos3UndPrefillExportModule:
    """Build UND-prefill from HF checkpoint via MR weight split."""
    with open(os.path.join(checkpoint, "transformer", "config.json")) as f:
        tcfg = json.load(f)

    und_weights, _ = split_transformer_weights(str(Path(checkpoint) / "transformer"))
    cfg = und_config_from_transformer(tcfg)
    module = Cosmos3UndPrefillExportModule(cfg).to(dtype)
    load_und_weights(module, und_weights, dtype)
    return module.eval()


def load_policy_gen(
    checkpoint: str,
    dtype: torch.dtype,
    *,
    action_chunk_size: int | None = None,
    num_frames: int | None = None,
) -> Cosmos3GenStepExportModule:
    """Build GEN step from HF checkpoint via MR weight split / domain bake."""
    with open(os.path.join(str(checkpoint), "transformer", "config.json")) as f:
        tcfg = json.load(f)
    _, gen_weights = split_transformer_weights(str(Path(checkpoint) / "transformer"))
    cfg = gen_config_from_transformer(
        tcfg, action_chunk_size=action_chunk_size, num_frames=num_frames
    )
    module = Cosmos3GenStepExportModule(cfg).to(dtype)
    load_gen_weights(module, gen_weights, dtype)
    return bind_named_gen_forward(module.eval())

def und_prefill_io_names(num_layers: int) -> tuple[list[str], list[str]]:
    inputs = ["inputs_embeds", "rope_rotary_cos_sin", "attention_pos_id"]
    outputs = (
        [f"und_k_layer{i:02d}" for i in range(num_layers)]
        + [f"und_v_layer{i:02d}" for i in range(num_layers)]
        + ["hidden_states"]
    )
    return inputs, outputs


def gen_io_names(num_layers: int) -> tuple[list[str], list[str]]:
    inputs = [
        "video_latent",
        "action_latent",
        "timestep",
        "token_noisy_mask",
        "action_noisy_mask",
        "rope_rotary_cos_sin",
        "attention_pos_id",
    ] + [f"und_k_layer{i:02d}" for i in range(num_layers)] + [
        f"und_v_layer{i:02d}" for i in range(num_layers)
    ]
    outputs = ["video_pred", "action_pred"]
    return inputs, outputs
