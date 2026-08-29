#!/usr/bin/env python3
"""Export Cosmos3-Edge policy engines for ``Cosmos3Runtime``.

Follows the same export flow as ``export_wfm_cosmos_edge.py`` (load → sample
inputs → stage GPU → compile components), but writes the UND-KV-cached policy
layout expected by ``Cosmos3Runtime``::

    <engine_dir>/
      und_prefill/{und_prefill.engine,config.json}
      gen/{gen.engine,config.json}
      vae_encoder/{vae_encoder.engine,config.json}
      text_tokenizer/
      embed_tokens.safetensors

Rebuilds standalone UND/GEN module trees from the split checkpoint (not live MoT).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_TEST_ROOT = Path(__file__).resolve().parents[4]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

# This directory name is not a valid Python identifier, so siblings are imported
# as top-level modules rather than as a package.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch
import torch_tensorrt
from huggingface_hub import snapshot_download

if hasattr(getattr(torch_tensorrt, "logging", None), "set_level"):
    torch_tensorrt.logging.set_level(logging.WARNING)

from trt.compile import make_input_spec, save_trt_engine_module
from trt.measure import parity
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import configure_thor_pytorch, free_cuda_memory
from config import (
    make_gen_config,
    make_und_prefill_config,
    make_vae_encoder_config,
)
from policy import (
    Cosmos3VaeEncoderExportModule,
    gen_io_names,
    load_policy_gen,
    load_policy_und_prefill,
    und_prefill_io_names,
)
from weights import split_transformer_weights

configure_thor_pytorch()

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

TRT_SETTINGS_EXPORT = {**TRT_SETTINGS, "offload_module_to_cpu": False}

# Both MoT towers build with fp16 matmul accumulation and TensorRT's fused
# attention, matching the Edge-LLM ONNX builder. The VAE encoder keeps the base
# settings because fused SDPA there trips a TensorRT Myelin SSA check.
MOT_TRT_SETTINGS_EXPORT = {
    **TRT_SETTINGS_EXPORT,
    "use_fp32_acc": False,
    "decompose_attention": False,
}


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
    """Packed text-only mRoPE cos|sin [B,S,D] and iota position ids [B,S]."""
    half = head_dim // 2
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (
        rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    angles = pos[:, None] * freqs[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rope = torch.cat([cos, sin], dim=-1)[None].expand(batch, seq_len, head_dim).contiguous()
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32)[None].expand(batch, seq_len).contiguous()
    return rope, pos_ids


def _stage_tokenizer_and_embed(checkpoint: Path, engine_root: Path, dtype: torch.dtype) -> None:
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
    und_weights, _ = split_transformer_weights(str(checkpoint / "transformer"))
    if "embed_tokens.weight" in und_weights:
        save_file(
            {"embed_tokens.weight": und_weights["embed_tokens.weight"].to(dtype).contiguous()},
            engine_root / "embed_tokens.safetensors",
        )


def _compile_parity(
    module: torch.nn.Module,
    sample_inputs: tuple,
    *,
    trt_settings: dict | None = None,
) -> torch.nn.Module:
    exported = torch.export.export(module, args=sample_inputs, strict=False)
    return torch_tensorrt.dynamo.compile(
        exported,
        inputs=make_input_spec(sample_inputs),
        **(trt_settings or TRT_SETTINGS_EXPORT),
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

    # ------------------------------------------------------------------ VAE
    print("[2] Export vae_encoder")
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        checkpoint, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()

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
    parity("policy VAE A vs C", cond_eager, cond_trt)

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
    free_cuda_memory(vae_trt, vae_module, vae, cond_trt, cond_eager)
    _sync_gpu()
    print(f"  cond_latent: {tuple(cond_latent.shape)}")
    print(f"  vae eager/trt: {vae_eager_ms:.3f} / {vae_trt_ms:.3f} ms  ({_speedup(vae_eager_ms, vae_trt_ms)})")

    # -------------------------------------------------------------- UND prefill
    print("[3] Export und_prefill (frozen K/V)")
    und = load_policy_und_prefill(str(checkpoint), dtype).to(device)
    cfg = und.cfg
    # Torch-TRT is static-shaped; must match tokenized prompt length at inference.
    und_len = int(os.environ.get("COSMOS3_UND_LEN", args.und_len))
    batch = 1
    head_dim = int(cfg["head_dim"])
    n_layers = int(cfg["num_hidden_layers"])
    n_kv = int(cfg["num_key_value_heads"])

    und_weights, _ = split_transformer_weights(str(checkpoint / "transformer"))
    embed = und_weights["embed_tokens.weight"].to(device=device, dtype=dtype)
    input_ids = torch.randint(0, embed.shape[0], (batch, und_len), device=device)
    inputs_embeds = embed[input_ids].contiguous()
    rope, pos_ids = _make_text_rope(
        batch=batch,
        seq_len=und_len,
        head_dim=head_dim,
        rope_theta=float(cfg["rope_theta"]),
        device=device,
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
    parity("policy UND k0 A vs C", und_out_eager[0], und_out_trt[0])
    parity("policy UND hidden A vs C", und_out_eager[-1], und_out_trt[-1])

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
    free_cuda_memory(und_trt, und, und_out_trt, und_out_eager[-1], embed, input_ids)
    _sync_gpu()
    print(f"  und_len={und_len} layers={n_layers} kv={n_kv}x{head_dim}")
    print(f"  und eager/trt: {und_eager_ms:.3f} / {und_trt_ms:.3f} ms  ({_speedup(und_eager_ms, und_trt_ms)})")

    # ------------------------------------------------------------------- GEN
    print("[4] Export gen (cross-attend frozen UND KV)")
    _, c_lat, t_lat, h_lat, w_lat = cond_latent.shape
    gen = load_policy_gen(
        str(checkpoint),
        dtype,
        action_chunk_size=args.action_chunk_size,
        num_frames=args.num_frames,
    ).to(device)
    gen.cfg.latent_channel = int(c_lat)
    gen.cfg.latent_t = int(t_lat)
    gen.cfg.latent_h = int(h_lat)
    gen.cfg.latent_w = int(w_lat)

    action_len = int(gen.cfg.action_chunk_size)
    max_action_dim = int(gen.cfg.max_action_dim)
    num_video_tokens = int(gen.cfg.num_video_tokens)
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
    patches_per_frame = (int(h_lat) // gen.cfg.latent_patch_size) * (int(w_lat) // gen.cfg.latent_patch_size)
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

    gen_trt = _compile_parity(gen, gen_inputs, trt_settings=MOT_TRT_SETTINGS_EXPORT)
    with torch.no_grad():
        video_pred_t, action_pred_t = gen_trt(*gen_inputs)
    gen_trt_ms = _time_cuda_ms(lambda: gen_trt(*gen_inputs), device=device)
    parity("policy GEN video A vs C", video_pred_e, video_pred_t)
    parity("policy GEN action A vs C", action_pred_e, action_pred_t)

    gen_in_names, gen_out_names = gen_io_names(n_layers)
    if not args.skip_save:
        with open(checkpoint / "transformer" / "config.json") as f:
            tcfg = json.load(f)
        gen_cfg_json = make_gen_config(
            gen.cfg, tcfg, max_und_len=max(und_len, 512), fps=args.fps
        )
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
            extra_config=gen_cfg_json,
            trt_settings=MOT_TRT_SETTINGS_EXPORT,
        )

        print("[5] Write tokenizer + embed_tokens")
        _stage_tokenizer_and_embed(checkpoint, engine_root, dtype)

    free_cuda_memory(gen_trt, gen, und_kv)
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
    parser = argparse.ArgumentParser(description="Export Cosmos3-Edge policy TRT engines for Cosmos3Runtime.")
    parser.add_argument("--engine-dir", required=True, help="Output directory for the engine bundle.")
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--prompt", default="Pick up the red cube.")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--timestep", type=float, default=500.0)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--und-len", type=int, default=121, help="Static UND sequence length baked into the engine.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Tensor dtype for export modules (fp16 matches current C++ runners).",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Parity only; do not write engines.",
    )
    args = parser.parse_args()
    export_engines(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
