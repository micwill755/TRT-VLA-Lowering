#!/usr/bin/env python3
"""Export Cosmos3-Edge policy engines for ``Cosmos3Runtime`` via live MoT wrap.

Follows the same export flow as ``export_wfm_cosmos_edge.py`` / the full-model
policy export (load → sample → compile → stage assets), writing::

    <engine_dir>/
      und_prefill/{und_prefill.engine,config.json}
      gen/{gen.engine,config.json}
      vae_encoder/{vae_encoder.engine,config.json}
      text_tokenizer/
      embed_tokens.safetensors

Reuses the live Diffusers ``transformer.layers`` (and GEN heads) in-place — no
MR module rebuild. Edge semantics Diffusers drops (reattached here):
- UND: no qk-norm; export ``k_norm_und_for_gen`` K for GEN context
- MLP: relu2(up) → down (checkpoint has no ``gate_proj``)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_TEST_ROOT = Path(__file__).resolve().parents[4]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

# ``edge-llm`` is not a valid Python identifier, so its modules (shared configs
# and export-module helpers) are imported as top-level modules by path.
for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / "edge-llm"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
from huggingface_hub import snapshot_download

if hasattr(getattr(torch_tensorrt, "logging", None), "set_level"):
    torch_tensorrt.logging.set_level(logging.WARNING)

from trt.compile import make_input_spec, save_trt_engine_module
from trt.measure import parity
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention, free_cuda_memory
from config import (
    DEFAULT_DOMAIN_ID,
    gen_config_from_transformer,
    make_gen_config,
    make_und_prefill_config,
    make_vae_encoder_config,
)
from edge_functions import TRT_SETTINGS, load_cosmos_from_pipeline
from policy import (
    Cosmos3VaeEncoderExportModule,
    apply_rope_packed,
    gen_io_names,
    und_prefill_io_names,
)

TRT_SETTINGS_EXPORT = {**TRT_SETTINGS, "offload_module_to_cpu": False}

# Both MoT towers build with fp16 matmul accumulation and TensorRT's fused
# attention, matching the Edge-LLM ONNX builder. The VAE encoder keeps the base
# settings because fused SDPA there trips a TensorRT Myelin SSA check.
MOT_TRT_SETTINGS_EXPORT = {
    **TRT_SETTINGS_EXPORT,
    "use_fp32_acc": False,
    "decompose_attention": False,
}


# ---------------------------------------------------------------------------
# Live-MoT UND/GEN wrappers (Cosmos3Runtime I/O; no MR rebuild)
# ---------------------------------------------------------------------------


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


def attach_k_norm_und_for_gen(transformer: nn.Module, checkpoint: str | Path, *, dtype: torch.dtype) -> None:
    """Load dropped ``k_norm_und_for_gen`` weights onto each MoT attention module."""
    from safetensors import safe_open

    ckpt = Path(checkpoint) / "transformer"
    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(ckpt.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if "k_norm_und_for_gen" in key:
                    weights[key] = f.get_tensor(key)

    eps = float(getattr(transformer.config, "rms_norm_eps", 1e-6))
    for i, layer in enumerate(transformer.layers):
        key = f"layers.{i}.self_attn.k_norm_und_for_gen.weight"
        if key not in weights:
            raise KeyError(f"Missing {key} in {ckpt}")
        dim = int(weights[key].shape[0])
        norm = _RMSNorm(dim, eps).to(dtype=dtype)
        with torch.no_grad():
            norm.weight.copy_(weights[key].to(dtype=dtype))
        layer.self_attn.k_norm_und_for_gen = norm


def _relu2_mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Edge MLP: ``down(relu(up(x))**2)`` — ignore fabricated ``gate_proj``."""
    return mlp.down_proj(F.relu(mlp.up_proj(x)).square())


class UndPrefillFromMoT(nn.Module):
    """UND-only prefill → per-layer ``und_k/v`` (+ final hidden). Cosmos3Runtime I/O."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.layers = transformer.layers
        self.norm = transformer.norm
        self.cfg = {
            "hidden_size": int(transformer.config.hidden_size),
            "num_hidden_layers": len(transformer.layers),
            "num_attention_heads": int(transformer.config.num_attention_heads),
            "num_key_value_heads": int(transformer.config.num_key_value_heads),
            "head_dim": int(transformer.config.head_dim),
            "rope_theta": float(transformer.config.rope_theta),
            "rms_norm_eps": float(getattr(transformer.config, "rms_norm_eps", 1e-6)),
            "use_und_k_norm_for_gen": True,
        }

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
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            b, s, _ = h.shape
            q = attn.to_q(h).view(b, s, attn.num_attention_heads, attn.head_dim).transpose(1, 2)
            k = attn.to_k(h).view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = attn.to_v(h).view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            q_rope = apply_rope_packed(q, rope_rotary_cos_sin, attention_pos_id)
            k_self = apply_rope_packed(k, rope_rotary_cos_sin, attention_pos_id)
            out = F.scaled_dot_product_attention(
                q_rope,
                k_self,
                v,
                is_causal=True,
                enable_gqa=True,
            )
            out = attn.to_out(out.transpose(1, 2).reshape(b, s, -1))
            x = x + out
            x = x + _relu2_mlp(layer.mlp, layer.post_attention_layernorm(x))

            k_gen_src = k
            if getattr(attn, "k_norm_und_for_gen", None) is not None:
                k_gen_src = attn.k_norm_und_for_gen(k)
            k_gen = apply_rope_packed(k_gen_src, rope_rotary_cos_sin, attention_pos_id)
            ks.append(k_gen.transpose(1, 2).contiguous())
            vs.append(v.transpose(1, 2).contiguous())

        return tuple(ks) + tuple(vs) + (self.norm(x),)


class GenStepFromMoT(nn.Module):
    """One GEN denoise step from live MoT + live heads. Cosmos3Runtime I/O."""

    def __init__(self, transformer: nn.Module, cfg: Any, *, domain_id: int = DEFAULT_DOMAIN_ID) -> None:
        super().__init__()
        if not getattr(transformer, "action_gen", False):
            raise RuntimeError("transformer.action_gen must be True for policy GEN export")
        self.cfg = cfg
        self.layers = transformer.layers
        self.norm_moe_gen = transformer.norm_moe_gen
        self.proj_in = transformer.proj_in
        self.proj_out = transformer.proj_out
        self.time_proj = transformer.time_proj
        self.time_embedder = transformer.time_embedder
        self.action_modality_embed = transformer.action_modality_embed

        # Bake embodiment domain into plain matmul params (Cosmos3Runtime contract).
        with torch.no_grad():
            ain = transformer.action_proj_in
            aout = transformer.action_proj_out
            in_w = ain.fc.weight[domain_id].view(ain.input_size, ain.output_size).detach().clone()
            in_b = ain.bias.weight[domain_id].detach().clone()
            out_w = aout.fc.weight[domain_id].view(aout.input_size, aout.output_size).detach().clone()
            out_b = aout.bias.weight[domain_id].detach().clone()
        self.action_in_weight = nn.Parameter(in_w)
        self.action_in_bias = nn.Parameter(in_b)
        self.action_out_weight = nn.Parameter(out_w)
        self.action_out_bias = nn.Parameter(out_b)

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

        t_scaled = timestep.float() * float(cfg.timestep_scale)
        t_embed = self.time_embedder(self.time_proj(t_scaled)).to(io_type)
        video_tokens = video_tokens + t_embed[:, None, :] * token_noisy_mask.to(io_type)
        action_tokens = action_tokens + t_embed[:, None, :] * action_noisy_mask.to(io_type)
        hidden = torch.cat([video_tokens, action_tokens], dim=1)

        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = layer.input_layernorm_moe_gen(hidden)
            b, s, _ = h.shape
            q = attn.add_q_proj(h).view(b, s, attn.num_attention_heads, attn.head_dim).transpose(1, 2)
            k = attn.add_k_proj(h).view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = attn.add_v_proj(h).view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            q = attn.norm_added_q(q)
            k = attn.norm_added_k(k)
            q = apply_rope_packed(q, rope_rotary_cos_sin, attention_pos_id)
            k = apply_rope_packed(k, rope_rotary_cos_sin, attention_pos_id)
            k_all = torch.cat([k_und[i].transpose(1, 2).to(k.dtype), k], dim=2)
            v_all = torch.cat([v_und[i].transpose(1, 2).to(v.dtype), v], dim=2)
            out = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                is_causal=False,
                enable_gqa=True,
            )
            out = attn.to_add_out(out.transpose(1, 2).reshape(b, s, -1))
            hidden = hidden + out
            hidden = hidden + _relu2_mlp(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(hidden))

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


def bind_named_gen_forward(module: GenStepFromMoT) -> GenStepFromMoT:
    """Name GEN UND-KV inputs ``und_k_layerXX`` / ``und_v_layerXX`` for Cosmos3Runtime."""
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
    exec(src, ns)  # noqa: S102
    module.forward = ns["forward"].__get__(module, type(module))  # type: ignore[method-assign]
    return module


def _parse_dtype(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}; use bf16 or fp16")


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _time_cuda_ms(fn, *, device: torch.device, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _speedup(eager_ms: float, trt_ms: float) -> str:
    if eager_ms <= 0.0 or trt_ms <= 0.0:
        return "n/a"
    return f"{eager_ms / trt_ms:.3f}x"


def _make_text_rope(
    *,
    batch: int,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (
        rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    angles = pos[:, None] * freqs[None, :]
    rope = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)[None].expand(batch, seq_len, head_dim).contiguous()
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32)[None].expand(batch, seq_len).contiguous()
    return rope, pos_ids


def _stage_tokenizer_and_embed(checkpoint: Path, engine_root: Path, transformer, dtype: torch.dtype) -> None:
    from safetensors.torch import save_file

    tok_src = checkpoint / "text_tokenizer"
    tok_dst = engine_root / "text_tokenizer"
    if tok_src.is_dir():
        shutil.copytree(tok_src, tok_dst, dirs_exist_ok=True)
    # C++ tokenizer requires processed_chat_template.json (not shipped in HF Edge).
    template_dst = tok_dst / "processed_chat_template.json"
    if not template_dst.is_file():
        fallback = Path("/tmp/cosmos3_policy_engines/text_tokenizer/processed_chat_template.json")
        if fallback.is_file():
            shutil.copy2(fallback, template_dst)
        else:
            template_dst.write_text(
                json.dumps(
                    {
                        "model_path": str(tok_src),
                        "roles": {
                            "system": {"prefix": "", "suffix": "\n"},
                            "user": {"prefix": "User: ", "suffix": "\n"},
                            "assistant": {"prefix": "Assistant: ", "suffix": "\n"},
                        },
                        "content_types": {},
                        "generation_prompt": "Assistant: ",
                        "default_system_prompt": "",
                    },
                    indent=2,
                )
                + "\n"
            )
    weight = transformer.embed_tokens.weight.detach().to(dtype).contiguous().cpu()
    save_file({"embed_tokens.weight": weight}, engine_root / "embed_tokens.safetensors")


def _compile_parity(
    module: torch.nn.Module,
    sample_inputs: tuple,
    *,
    trt_settings: dict[str, Any] | None = None,
) -> torch.nn.Module:
    settings = trt_settings or TRT_SETTINGS_EXPORT
    exported = torch.export.export(module, args=sample_inputs, strict=False)
    return torch_tensorrt.dynamo.compile(
        exported,
        inputs=make_input_spec(sample_inputs),
        **settings,
    )


def export_engines(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Cosmos3 policy TRT export.")

    dtype = _parse_dtype(args.dtype)
    engine_root = Path(args.engine_dir).resolve()
    engine_root.mkdir(parents=True, exist_ok=True)

    load_plugins_for_trt()
    print(f"[1] Load {args.model_id} (export dtype={dtype})")
    checkpoint = Path(snapshot_download(args.model_id))
    components = load_cosmos_from_pipeline(args.model_id, dtype=dtype, device=device)
    transformer = components["transformer"]
    vae = components["vae"]
    force_hf_attention(transformer, "eager")

    transformer.to("cpu")
    attach_k_norm_und_for_gen(transformer, checkpoint, dtype=dtype)
    transformer.to(device).eval()

    # ------------------------------------------------------------------ VAE
    print("[2] Export vae_encoder")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pixels = torch.randn(
        1, 3, args.num_frames, args.height, args.width,
        device=device, dtype=torch.float32, generator=generator,
    ).tanh()
    vae_module = Cosmos3VaeEncoderExportModule(vae).eval().to(device)
    with torch.no_grad():
        cond_eager = vae_module(pixels)
    vae_eager_ms = _time_cuda_ms(lambda: vae_module(pixels), device=device)
    vae_trt = _compile_parity(vae_module, (pixels,))
    with torch.no_grad():
        cond_trt = vae_trt(pixels)
    vae_trt_ms = _time_cuda_ms(lambda: vae_trt(pixels), device=device)
    parity("live-MoT VAE A vs C", cond_eager, cond_trt)

    if not args.skip_save:
        print(f"  compiling vae_encoder -> {engine_root / 'vae_encoder' / 'vae_encoder.engine'}")
        save_trt_engine_module(
            vae_module,
            (pixels,),
            engine_root / "vae_encoder",
            engine_file="vae_encoder.engine",
            model_type="cosmos3_vae_encoder",
            component="vae_encoder",
            input_names=["pixel_values"],
            output_names=["cond_latent"],
            extra_config=make_vae_encoder_config(args.height, args.width, args.num_frames),
            trt_settings=TRT_SETTINGS_EXPORT,
        )

    cond_latent = cond_eager.detach().clone()
    free_cuda_memory(vae_trt, vae_module, cond_trt, cond_eager)
    _sync_gpu()
    print(f"  cond_latent: {tuple(cond_latent.shape)}")
    print(f"  vae eager/trt: {vae_eager_ms:.3f} / {vae_trt_ms:.3f} ms  ({_speedup(vae_eager_ms, vae_trt_ms)})")

    # -------------------------------------------------------------- UND prefill
    print("[3] Export und_prefill (frozen K/V from live MoT)")
    und = UndPrefillFromMoT(transformer).eval()
    cfg = und.cfg
    und_len = int(os.environ.get("COSMOS3_UND_LEN", args.und_len))
    batch = 1
    head_dim = int(cfg["head_dim"])
    n_layers = int(cfg["num_hidden_layers"])
    n_kv = int(cfg["num_key_value_heads"])

    embed = transformer.embed_tokens.weight.detach().to(device=device, dtype=dtype)
    input_ids = torch.randint(0, embed.shape[0], (batch, und_len), device=device)
    inputs_embeds = embed[input_ids].contiguous()
    rope, pos_ids = _make_text_rope(
        batch=batch, seq_len=und_len, head_dim=head_dim, rope_theta=float(cfg["rope_theta"]), device=device
    )
    rope = rope.to(torch.float32)
    und_inputs = (inputs_embeds, rope, pos_ids)

    with torch.no_grad():
        und_out_eager = und(*und_inputs)
    und_eager_ms = _time_cuda_ms(lambda: und(*und_inputs), device=device)
    und_trt = _compile_parity(und, und_inputs, trt_settings=MOT_TRT_SETTINGS_EXPORT)
    with torch.no_grad():
        und_out_trt = und_trt(*und_inputs)
    und_trt_ms = _time_cuda_ms(lambda: und_trt(*und_inputs), device=device)
    parity("live-MoT UND k0 A vs C", und_out_eager[0], und_out_trt[0])
    parity("live-MoT UND hidden A vs C", und_out_eager[-1], und_out_trt[-1])

    und_in_names, und_out_names = und_prefill_io_names(n_layers)
    if not args.skip_save:
        print(f"  compiling und_prefill -> {engine_root / 'und_prefill' / 'und_prefill.engine'}")
        save_trt_engine_module(
            und,
            und_inputs,
            engine_root / "und_prefill",
            engine_file="und_prefill.engine",
            model_type="cosmos3_und_prefill",
            component="und_prefill",
            input_names=und_in_names,
            output_names=und_out_names,
            extra_config=make_und_prefill_config(cfg, max_und_len=max(und_len, 512)),
            trt_settings=MOT_TRT_SETTINGS_EXPORT,
        )

    und_kv = tuple(t.detach().contiguous() for t in und_out_eager[:-1])
    free_cuda_memory(und_trt, und_out_trt, und_out_eager[-1], embed, input_ids)
    _sync_gpu()
    print(f"  und_len={und_len} layers={n_layers} kv={n_kv}x{head_dim}")
    print(f"  und eager/trt: {und_eager_ms:.3f} / {und_trt_ms:.3f} ms  ({_speedup(und_eager_ms, und_trt_ms)})")

    # ------------------------------------------------------------------- GEN
    print("[4] Export gen (live MoT + heads, frozen UND KV)")
    with open(checkpoint / "transformer" / "config.json") as f:
        tcfg = json.load(f)
    gen_cfg = gen_config_from_transformer(
        tcfg, action_chunk_size=args.action_chunk_size, num_frames=args.num_frames
    )
    _, c_lat, t_lat, h_lat, w_lat = cond_latent.shape
    gen_cfg.latent_channel = int(c_lat)
    gen_cfg.latent_t = int(t_lat)
    gen_cfg.latent_h = int(h_lat)
    gen_cfg.latent_w = int(w_lat)

    gen = bind_named_gen_forward(GenStepFromMoT(transformer, gen_cfg).eval())
    assert gen.layers[0] is transformer.layers[0]
    print("  GEN wrapper reuses live Diffusers MoT layers + heads")

    action_len = int(gen_cfg.action_chunk_size)
    max_action_dim = int(gen_cfg.max_action_dim)
    num_video_tokens = int(gen_cfg.num_video_tokens)
    gen_len = num_video_tokens + action_len

    video_latent = torch.randn(
        batch, int(c_lat), int(t_lat), int(h_lat), int(w_lat),
        device=device, dtype=torch.float32, generator=generator,
    )
    video_latent[:, :, 0].copy_(cond_latent[:, :, 0])
    action_latent = torch.randn(
        batch, action_len, max_action_dim, device=device, dtype=torch.float32, generator=generator
    )
    timestep = torch.full((batch,), args.timestep, device=device, dtype=torch.float32)
    token_noisy_mask = torch.ones(batch, num_video_tokens, 1, device=device, dtype=torch.float32)
    patches_per_frame = (int(h_lat) // gen_cfg.latent_patch_size) * (int(w_lat) // gen_cfg.latent_patch_size)
    token_noisy_mask[:, :patches_per_frame] = 0.0
    action_noisy_mask = torch.ones(batch, action_len, 1, device=device, dtype=torch.float32)
    gen_rope, gen_pos = _make_text_rope(
        batch=batch, seq_len=gen_len, head_dim=head_dim, rope_theta=float(cfg["rope_theta"]), device=device
    )
    gen_rope = gen_rope.to(torch.float32)
    gen_inputs = (
        video_latent,
        action_latent,
        timestep,
        token_noisy_mask,
        action_noisy_mask,
        gen_rope,
        gen_pos,
        *und_kv,
    )

    with torch.no_grad():
        video_pred_e, action_pred_e = gen(*gen_inputs)
    gen_eager_ms = _time_cuda_ms(lambda: gen(*gen_inputs), device=device)
    gen_trt = _compile_parity(
        gen,
        gen_inputs,
        trt_settings=MOT_TRT_SETTINGS_EXPORT,
    )
    with torch.no_grad():
        video_pred_t, action_pred_t = gen_trt(*gen_inputs)
    gen_trt_ms = _time_cuda_ms(lambda: gen_trt(*gen_inputs), device=device)
    parity("live-MoT GEN video A vs C", video_pred_e, video_pred_t)
    parity("live-MoT GEN action A vs C", action_pred_e, action_pred_t)

    gen_in_names, gen_out_names = gen_io_names(n_layers)
    if not args.skip_save:
        print(f"  compiling gen -> {engine_root / 'gen' / 'gen.engine'}")
        save_trt_engine_module(
            gen,
            gen_inputs,
            engine_root / "gen",
            engine_file="gen.engine",
            model_type="cosmos3_gen",
            component="gen",
            input_names=gen_in_names,
            output_names=gen_out_names,
            extra_config=make_gen_config(gen_cfg, tcfg, max_und_len=max(und_len, 512), fps=args.fps),
            trt_settings=MOT_TRT_SETTINGS_EXPORT,
        )
        print("[5] Write tokenizer + embed_tokens")
        _stage_tokenizer_and_embed(checkpoint, engine_root, transformer, dtype)

    free_cuda_memory(gen_trt, und_kv)
    _sync_gpu()

    total_e = vae_eager_ms + und_eager_ms + gen_eager_ms
    total_t = vae_trt_ms + und_trt_ms + gen_trt_ms
    print(f"\nExport complete: {engine_root}")
    print(f"  vae  eager/trt: {vae_eager_ms:.3f} / {vae_trt_ms:.3f} ms  ({_speedup(vae_eager_ms, vae_trt_ms)})")
    print(f"  und  eager/trt: {und_eager_ms:.3f} / {und_trt_ms:.3f} ms  ({_speedup(und_eager_ms, und_trt_ms)})")
    print(f"  gen  eager/trt: {gen_eager_ms:.3f} / {gen_trt_ms:.3f} ms  ({_speedup(gen_eager_ms, gen_trt_ms)})")
    print(f"  total eager/trt: {total_e:.3f} / {total_t:.3f} ms  ({_speedup(total_e, total_t)})")
    if not args.skip_save:
        print("  Test with:")
        print(f"    cosmos3_policy_inference --engineDir {engine_root} \\")
        print(f'      --image <image> --prompt "{args.prompt}" --output action.json')
    return engine_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Cosmos3-Edge split-MoT policy engines for Cosmos3Runtime."
    )
    parser.add_argument("--engine-dir", required=True, help="Output directory for the engine bundle.")
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--prompt", default="Pick up the red cube.")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--timestep", type=float, default=500.0)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--und-len", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Tensor dtype for export modules (fp16 matches current C++ runners).",
    )
    parser.add_argument("--skip-save", action="store_true", help="Parity only; do not write engines.")
    args = parser.parse_args()
    export_engines(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
