#!/usr/bin/env python3
"""Export Cosmos3-Edge WFM TensorRT engines for ``WFMInferenceRuntime``.

Writes the on-disk layout expected by TensorRT-Edge-LLM::

    <engine_dir>/
      config.json
      packing_static.json
      embedding.safetensors
      tokenizer/{tokenizer.json, processed_chat_template.json, ...}
      visual_encode/{config.json, visual_encode.engine}
      visual_decode/{config.json, visual_decode.engine}
      embed/{config.json, embed.engine}
      mot_backbone/{config.json, mot_backbone.engine}
      denoise_head/{config.json, denoise_head.engine}

Edge is vision-only (2-input embed, no sound engines). The ~3.9B-parameter MoT
backbone (~7 GB bf16) fits on a 32 GB GPU during export.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch_tensorrt
from accelerate.utils import set_module_tensor_to_device
from diffusers import Cosmos3OmniPipeline

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[4]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.compile import save_trt_engine_module
from trt.modules.cosmos.backbone import Cosmos3MoTBackboneExportModule
from trt.modules.cosmos.decode import CosmosVaeDecodeExportModule
from trt.modules.cosmos.embed import Cosmos3VisionGenEmbedExportModule
from trt.modules.cosmos.head import Cosmos3VisionDenoiseHeadExportModule
from trt.modules.cosmos.packing import (
    build_cosmos_packed_static,
    build_wfm_root_config,
    save_wfm_tokenizer_assets,
    serialize_cosmos_packed_static,
)
from trt.modules.cosmos.vision import CosmosVaeEncodeExportModule
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

TRT_SETTINGS_EXPORT = {**TRT_SETTINGS, "offload_module_to_cpu": True}


def fix_edge_diffusers_weights(
    transformer,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Materialize Edge checkpoint tensors that diffusers leaves on meta."""
    named = dict(transformer.named_parameters())
    for name, param in list(transformer.named_parameters()):
        if not param.is_meta:
            continue
        if name.endswith("gate_proj.weight"):
            value = named[name.replace("gate_proj.weight", "up_proj.weight")].detach().clone()
        elif name.endswith(("norm_q.weight", "norm_k.weight")):
            value = torch.ones(param.shape, dtype=dtype)
        else:
            value = torch.zeros(param.shape, dtype=dtype)
        set_module_tensor_to_device(transformer, name, device=device, value=value, dtype=dtype)


def load_cosmos_from_pipeline(
    model_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cuda",
) -> dict:
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        enable_safety_checker=False,
    )
    fix_edge_diffusers_weights(pipe.transformer, device="cpu", dtype=dtype)
    device = torch.device(device)
    return {
        "transformer": pipe.transformer.to(device).eval(),
        "vae": pipe.vae.to(device).eval(),
        "tokenizer": pipe.text_tokenizer,
        "scheduler": pipe.scheduler,
    }


@torch.no_grad()
def build_und_seq(transformer, input_ids: torch.Tensor) -> torch.Tensor:
    return transformer.embed_tokens(input_ids)


@torch.no_grad()
def build_rotary_emb(
    transformer,
    position_ids: torch.Tensor,
    und_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cos, sin = transformer.rotary_emb(
        position_ids=position_ids.unsqueeze(1),
        device=device,
        dtype=dtype,
    )
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    return (
        cos[:und_len].contiguous(),
        sin[:und_len].contiguous(),
        cos[und_len:].contiguous(),
        sin[und_len:].contiguous(),
    )


@torch.no_grad()
def eager_vision_gen_embed(
    transformer,
    packed_static: dict,
    vision_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    packed_tokens, _ = transformer._patchify_and_pack_latents([vision_latents])
    packed_tokens = transformer.proj_in(packed_tokens)
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=vision_latents.device, dtype=vision_latents.dtype)
    timestep = timestep.to(device=vision_latents.device, dtype=vision_latents.dtype).reshape(())
    timesteps = timestep.expand(int(packed_static["num_noisy_tokens"])) * transformer.config.timestep_scale
    timestep_embeds = transformer.time_embedder(transformer.time_proj(timesteps)).to(packed_tokens.dtype)
    return transformer._apply_timestep_embeds_to_noisy_tokens(
        packed_tokens=packed_tokens,
        packed_timestep_embeds=timestep_embeds,
        noisy_frame_indexes=packed_static["vision_noisy_frame_indexes"],
        token_shapes=packed_static["vision_token_shapes"],
    )


@torch.no_grad()
def eager_mot_backbone(
    transformer,
    und_seq: torch.Tensor,
    gen_seq: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos_und, sin_und, cos_gen, sin_gen = rotary_emb
    for layer in transformer.layers:
        und_seq, gen_seq = layer(und_seq, gen_seq, (cos_und, sin_und, cos_gen, sin_gen))
    return torch.cat([transformer.norm(und_seq), transformer.norm_moe_gen(gen_seq)], dim=0)


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _parse_dtype(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}; use bf16 or fp16")


def _save_embedding_table(transformer, output_path: Path, *, dtype: torch.dtype) -> None:
    from safetensors.torch import save_file

    weight = transformer.embed_tokens.weight.detach().cpu()
    if dtype == torch.float16:
        weight = weight.half()
    elif dtype == torch.bfloat16:
        weight = weight.to(torch.bfloat16)
    else:
        weight = weight.float()
    save_file({"embedding": weight}, output_path)


def _stage_vae_only_on_gpu(transformer, vae, device: torch.device) -> None:
    transformer.to("cpu")
    vae.to(device)
    _sync_gpu()


def _stage_transformer_only_on_gpu(transformer, vae, device: torch.device) -> None:
    vae.to("cpu")
    transformer.to(device)
    _sync_gpu()


def export_engines(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Cosmos Edge TRT export.")

    dtype = _parse_dtype(args.dtype)
    engine_root = Path(args.engine_dir).resolve()
    engine_root.mkdir(parents=True, exist_ok=True)

    load_plugins_for_trt()
    print(f"[1] Load {args.model_id} (export dtype={dtype})")
    components = load_cosmos_from_pipeline(args.model_id, dtype=dtype, device=device)
    transformer = components["transformer"]
    vae = components["vae"]
    tokenizer = components["tokenizer"]
    force_hf_attention(transformer, "eager")

    print("[2] Build packing metadata (vision-only)")
    packed_static, clean_latents, pixels = build_cosmos_packed_static(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        condition_frame_indexes=(0,),
        dtype=dtype,
    )

    und_seq = build_und_seq(transformer, packed_static["input_ids"])
    rotary_emb = build_rotary_emb(
        transformer,
        packed_static["position_ids"],
        int(packed_static["und_len"]),
        device=device,
        dtype=dtype,
    )
    timestep_t = torch.tensor(args.timestep, device=device, dtype=dtype)
    gen_seq = eager_vision_gen_embed(
        transformer,
        packed_static,
        clean_latents,
        timestep_t,
    )
    last_hidden = eager_mot_backbone(transformer, und_seq, gen_seq, rotary_emb)

    print("[3] Write root metadata + tokenizer")
    root_config = build_wfm_root_config(
        transformer=transformer,
        vae=vae,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        enable_sound=False,
        num_inference_steps=args.num_inference_steps,
    )
    (engine_root / "config.json").write_text(json.dumps(root_config, indent=2) + "\n")
    (engine_root / "packing_static.json").write_text(
        json.dumps(serialize_cosmos_packed_static(packed_static), indent=2) + "\n"
    )
    _save_embedding_table(transformer, engine_root / "embedding.safetensors", dtype=dtype)
    save_wfm_tokenizer_assets(engine_root, tokenizer=tokenizer)

    print("[4] Export visual_encode")
    _stage_vae_only_on_gpu(transformer, vae, device)
    visual_encode = CosmosVaeEncodeExportModule(vae, pixels).eval().to(device=device, dtype=dtype)
    print(f"  compiling visual_encode -> {engine_root / 'visual_encode' / 'visual_encode.engine'}")
    save_trt_engine_module(
        visual_encode,
        (pixels,),
        engine_root / "visual_encode",
        engine_file="visual_encode.engine",
        model_type="cosmos_visual_encode",
        component="visual_encode",
        input_names=["pixels"],
        output_names=["latents"],
        trt_settings=TRT_SETTINGS_EXPORT,
    )
    del visual_encode
    _sync_gpu()

    print("[5] Export embed (vision -> gen_seq)")
    _stage_transformer_only_on_gpu(transformer, vae, device)
    embed_module = Cosmos3VisionGenEmbedExportModule(
        transformer,
        packed_static=packed_static,
        sample_latents=clean_latents,
        sample_timestep=args.timestep,
    ).eval().to(device=device, dtype=dtype)
    print(f"  compiling embed -> {engine_root / 'embed' / 'embed.engine'}")
    save_trt_engine_module(
        embed_module,
        (clean_latents, timestep_t),
        engine_root / "embed",
        engine_file="embed.engine",
        model_type="cosmos_embed",
        component="embed",
        input_names=["vision_latents", "timestep"],
        output_names=["gen_seq"],
        trt_settings=TRT_SETTINGS_EXPORT,
    )
    del embed_module
    _sync_gpu()

    if args.skip_backbone:
        print("[6] Skip mot_backbone (--skip-backbone)")
    else:
        print("[6] Export mot_backbone (28 layers, ~7 GB bf16)")
        _stage_transformer_only_on_gpu(transformer, vae, device)
        backbone_module = Cosmos3MoTBackboneExportModule(
            transformer,
            sample_und_seq=und_seq,
            sample_gen_seq=gen_seq,
            sample_rotary_emb=rotary_emb,
        ).eval().to(device=device, dtype=dtype)
        print(f"  compiling mot_backbone -> {engine_root / 'mot_backbone' / 'mot_backbone.engine'}")
        save_trt_engine_module(
            backbone_module,
            (und_seq, gen_seq, *rotary_emb),
            engine_root / "mot_backbone",
            engine_file="mot_backbone.engine",
            model_type="cosmos_mot_backbone",
            component="mot_backbone",
            input_names=["und_seq", "gen_seq", "cos_und", "sin_und", "cos_gen", "sin_gen"],
            output_names=["last_hidden_state"],
            trt_settings=TRT_SETTINGS_EXPORT,
        )
        del backbone_module
        _sync_gpu()

    print("[7] Export denoise_head (vision)")
    _stage_transformer_only_on_gpu(transformer, vae, device)
    vision_head = Cosmos3VisionDenoiseHeadExportModule(
        transformer,
        packed_static=packed_static,
        sample_latents=clean_latents,
        sample_last_hidden=last_hidden,
    ).eval().to(device=device, dtype=dtype)
    print(f"  compiling denoise_head -> {engine_root / 'denoise_head' / 'denoise_head.engine'}")
    save_trt_engine_module(
        vision_head,
        (last_hidden,),
        engine_root / "denoise_head",
        engine_file="denoise_head.engine",
        model_type="cosmos_denoise_head",
        component="denoise_head",
        input_names=["last_hidden_state"],
        output_names=["pred_vision_latents"],
        trt_settings=TRT_SETTINGS_EXPORT,
    )
    del vision_head, last_hidden, und_seq, gen_seq
    _sync_gpu()

    print("[8] Export visual_decode")
    _stage_vae_only_on_gpu(transformer, vae, device)
    visual_decode = CosmosVaeDecodeExportModule(vae, clean_latents).eval().to(device=device, dtype=dtype)
    print(f"  compiling visual_decode -> {engine_root / 'visual_decode' / 'visual_decode.engine'}")
    save_trt_engine_module(
        visual_decode,
        (clean_latents,),
        engine_root / "visual_decode",
        engine_file="visual_decode.engine",
        model_type="cosmos_visual_decode",
        component="visual_decode",
        input_names=["latents"],
        output_names=["pixels"],
        trt_settings=TRT_SETTINGS_EXPORT,
    )
    del visual_decode
    _sync_gpu()

    print(f"\nExport complete: {engine_root}")
    if args.skip_backbone:
        print("  Note: mot_backbone/ was not exported. C++ denoise requires it.")
    print("  Test with:")
    print(f"    wfm_inference --engineDir={engine_root} \\")
    print("      --inputFile=examples/wfm/wfm_input_example.json \\")
    print("      --outputFile=/tmp/wfm_output.json")
    print("  Use the same prompt as export for matching packing_static und_len.")
    return engine_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Cosmos3-Edge WFM TRT engines for Edge-LLM runtime.")
    parser.add_argument("--engine-dir", required=True, help="Output directory for the engine bundle.")
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--prompt", default="A robot arm picks up a red cube.")
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--num-inference-steps", type=int, default=2, help="Written to config.json (use 2 for smoke tests).")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Tensor dtype for export modules (fp16 matches current C++ runners).",
    )
    parser.add_argument(
        "--skip-backbone",
        action="store_true",
        help="Skip mot_backbone export (denoise path unavailable in C++ runtime).",
    )
    args = parser.parse_args()
    export_engines(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
