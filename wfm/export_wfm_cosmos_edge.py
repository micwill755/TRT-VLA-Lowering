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

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[1]
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

from wfm.test_wfm_cosmos_edge_e2e import (
    TRT_SETTINGS,
    build_rotary_emb,
    build_und_seq,
    eager_mot_backbone,
    eager_vision_gen_embed,
    load_cosmos_from_pipeline,
)

TRT_SETTINGS_EXPORT = {**TRT_SETTINGS, "offload_module_to_cpu": True}


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


def _export_component(
    module: torch.nn.Module,
    sample_inputs: tuple,
    engine_dir: Path,
    *,
    engine_file: str,
    model_type: str,
    component: str,
    input_names: list[str],
    output_names: list[str],
) -> Path:
    engine_dir.mkdir(parents=True, exist_ok=True)
    print(f"  compiling {component} -> {engine_dir / engine_file}")
    return save_trt_engine_module(
        module,
        sample_inputs,
        engine_dir,
        engine_file=engine_file,
        model_type=model_type,
        component=component,
        input_names=input_names,
        output_names=output_names,
        trt_settings=TRT_SETTINGS_EXPORT,
    )


def export_engines(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Cosmos Edge TRT export.")

    dtype = _parse_dtype(args.dtype)
    engine_root = Path(args.engine_dir).resolve()
    engine_root.mkdir(parents=True, exist_ok=True)

    load_plugins_for_trt()
    print(f"[1] Load {args.model_id} (export dtype={dtype})")
    components = load_cosmos_from_pipeline(dtype=dtype, device=device)
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
    _export_component(
        visual_encode,
        (pixels,),
        engine_root / "visual_encode",
        engine_file="visual_encode.engine",
        model_type="cosmos_visual_encode",
        component="visual_encode",
        input_names=["pixels"],
        output_names=["latents"],
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
    _export_component(
        embed_module,
        (clean_latents, timestep_t),
        engine_root / "embed",
        engine_file="embed.engine",
        model_type="cosmos_embed",
        component="embed",
        input_names=["vision_latents", "timestep"],
        output_names=["gen_seq"],
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
        _export_component(
            backbone_module,
            (und_seq, gen_seq, *rotary_emb),
            engine_root / "mot_backbone",
            engine_file="mot_backbone.engine",
            model_type="cosmos_mot_backbone",
            component="mot_backbone",
            input_names=["und_seq", "gen_seq", "cos_und", "sin_und", "cos_gen", "sin_gen"],
            output_names=["last_hidden_state"],
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
    _export_component(
        vision_head,
        (last_hidden,),
        engine_root / "denoise_head",
        engine_file="denoise_head.engine",
        model_type="cosmos_denoise_head",
        component="denoise_head",
        input_names=["last_hidden_state"],
        output_names=["pred_vision_latents"],
    )
    del vision_head, last_hidden, und_seq, gen_seq
    _sync_gpu()

    print("[8] Export visual_decode")
    _stage_vae_only_on_gpu(transformer, vae, device)
    visual_decode = CosmosVaeDecodeExportModule(vae, clean_latents).eval().to(device=device, dtype=dtype)
    _export_component(
        visual_decode,
        (clean_latents,),
        engine_root / "visual_decode",
        engine_file="visual_decode.engine",
        model_type="cosmos_visual_decode",
        component="visual_decode",
        input_names=["latents"],
        output_names=["pixels"],
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
