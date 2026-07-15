#!/usr/bin/env python3
"""Export Cosmos3-Omni WFM TensorRT engines for ``WFMInferenceRuntime``.

Writes the on-disk layout expected by TensorRT-Edge-LLM::

    <engine_dir>/
      config.json
      packing_static.json
      embedding.safetensors
      visual_encode/{config.json, visual_encode.engine}
      visual_decode/{config.json, visual_decode.engine}
      embed/{config.json, embed.engine}
      mot_backbone/{config.json, mot_backbone.engine}   # optional (--export-backbone)
      denoise_head/{config.json, denoise_head.engine}
      denoise_head_sound/{config.json, denoise_head_sound.engine}
      audio_encode/{config.json, audio_encode.engine}
      audio_decode/{config.json, audio_decode.engine}

Export modules mirror ``test_wfm_cosmos_omni_e2e.py`` (conv-STFT encode, folded fp32 decode).
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
from trt.modules.cosmos.audio import CosmosAvaeDecodeTrtExportModule, CosmosAvaeEncodeConvStftExportModule
from trt.modules.cosmos.backbone import Cosmos3MoTBackboneExportModule
from trt.modules.cosmos.decode import CosmosVaeDecodeExportModule
from trt.modules.cosmos.embed import Cosmos3OmniGenEmbedExportModule
from trt.modules.cosmos.head import Cosmos3SoundDenoiseHeadExportModule, Cosmos3VisionDenoiseHeadExportModule
from trt.modules.cosmos.packing import (
    build_cosmos_omni_packed_static,
    build_wfm_root_config,
    serialize_cosmos_packed_static,
)
from trt.modules.cosmos.vision import CosmosVaeEncodeExportModule
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention

from wfm.test_wfm_cosmos_omni_e2e import (
    TRT_SETTINGS,
    build_rotary_emb,
    build_und_seq,
    eager_mot_backbone,
    eager_omni_gen_embed,
    load_cosmos_from_pipeline,
    release_sound_tokenizer_from_gpu,
    stage_sound_tokenizer_on_gpu,
    stage_transformer_only_on_gpu,
    stage_transformer_vae_on_gpu,
    stage_vae_only_on_gpu,
)

TRT_SETTINGS_AUDIO = {**TRT_SETTINGS, "offload_module_to_cpu": False}


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
    trt_settings: dict | None = None,
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
        trt_settings=trt_settings or TRT_SETTINGS,
    )


def export_engines(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Cosmos Omni TRT export.")

    dtype = _parse_dtype(args.dtype)
    engine_root = Path(args.engine_dir).resolve()
    engine_root.mkdir(parents=True, exist_ok=True)

    load_plugins_for_trt()
    print(f"[1] Load {args.model_id} (export dtype={dtype})")
    components = load_cosmos_from_pipeline(dtype=dtype)
    transformer = components["transformer"]
    vae = components["vae"]
    sound_tokenizer = components["sound_tokenizer"]
    tokenizer = components["tokenizer"]
    force_hf_attention(transformer, "eager")

    stage_transformer_vae_on_gpu(transformer, vae, device)

    print("[2] Build packing metadata (vision + sound)")
    packed_static, clean_latents, pixels, clean_sound, waveform = build_cosmos_omni_packed_static(
        transformer=transformer,
        vae=vae,
        sound_tokenizer=sound_tokenizer,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        condition_frame_indexes=(0,),
        dtype=dtype,
        enable_sound=True,
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
    gen_seq = eager_omni_gen_embed(
        transformer,
        packed_static,
        clean_latents,
        clean_sound,
        timestep_t,
    )
    last_hidden = eager_mot_backbone(transformer, und_seq, gen_seq, rotary_emb)

    print("[3] Write root metadata")
    root_config = build_wfm_root_config(
        transformer=transformer,
        vae=vae,
        sound_tokenizer=sound_tokenizer,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        enable_sound=True,
    )
    (engine_root / "config.json").write_text(json.dumps(root_config, indent=2) + "\n")
    (engine_root / "packing_static.json").write_text(
        json.dumps(serialize_cosmos_packed_static(packed_static), indent=2) + "\n"
    )
    _save_embedding_table(transformer, engine_root / "embedding.safetensors", dtype=dtype)

    print("[4] Export visual_encode")
    stage_vae_only_on_gpu(transformer, vae, device)
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

    print("[5] Export embed (Omni: vision + sound -> gen_seq)")
    stage_transformer_only_on_gpu(transformer, vae, device)
    embed_module = Cosmos3OmniGenEmbedExportModule(
        transformer,
        packed_static=packed_static,
        sample_vision_latents=clean_latents,
        sample_sound_latents=clean_sound,
        sample_timestep=args.timestep,
    ).eval().to(device=device, dtype=dtype)
    _export_component(
        embed_module,
        (clean_latents, clean_sound, timestep_t),
        engine_root / "embed",
        engine_file="embed.engine",
        model_type="cosmos_omni_embed",
        component="embed",
        input_names=["vision_latents", "sound_latents", "timestep"],
        output_names=["gen_seq"],
    )
    del embed_module
    _sync_gpu()

    if args.export_backbone:
        print("[6] Export mot_backbone (needs large GPU VRAM)")
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
    else:
        print("[6] Skip mot_backbone (pass --export-backbone on a 48GB+ GPU)")

    print("[7] Export denoise_head (vision)")
    stage_transformer_only_on_gpu(transformer, vae, device)
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
    del vision_head
    _sync_gpu()

    print("[8] Export denoise_head_sound")
    sound_head = Cosmos3SoundDenoiseHeadExportModule(
        transformer,
        packed_static=packed_static,
        sample_last_hidden=last_hidden,
    ).eval().to(device=device, dtype=dtype)
    _export_component(
        sound_head,
        (last_hidden,),
        engine_root / "denoise_head_sound",
        engine_file="denoise_head_sound.engine",
        model_type="cosmos_denoise_head_sound",
        component="denoise_head_sound",
        input_names=["last_hidden_state"],
        output_names=["pred_sound_latents"],
    )
    del sound_head, last_hidden, und_seq, gen_seq
    _sync_gpu()

    print("[9] Export visual_decode")
    stage_vae_only_on_gpu(transformer, vae, device)
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

    print("[10] Export audio_encode (conv-STFT) + audio_decode (folded fp32)")
    stage_sound_tokenizer_on_gpu(transformer, vae, sound_tokenizer, device)
    waveform_gpu = waveform.to(device=device, dtype=dtype)

    audio_encode = CosmosAvaeEncodeConvStftExportModule(sound_tokenizer, waveform_gpu).eval().to(device)
    _export_component(
        audio_encode,
        (waveform_gpu,),
        engine_root / "audio_encode",
        engine_file="audio_encode.engine",
        model_type="cosmos_audio_encode",
        component="audio_encode",
        input_names=["waveform"],
        output_names=["sound_latents"],
        trt_settings=TRT_SETTINGS_AUDIO,
    )
    del audio_encode

    sound_latents_batched = clean_sound.unsqueeze(0).to(device=device, dtype=dtype)
    audio_decode = CosmosAvaeDecodeTrtExportModule(sound_tokenizer, sound_latents_batched).eval().to(device)
    _export_component(
        audio_decode,
        (sound_latents_batched,),
        engine_root / "audio_decode",
        engine_file="audio_decode.engine",
        model_type="cosmos_audio_decode",
        component="audio_decode",
        input_names=["sound_latents"],
        output_names=["waveform"],
        trt_settings=TRT_SETTINGS_AUDIO,
    )
    del audio_decode
    release_sound_tokenizer_from_gpu(sound_tokenizer)
    _sync_gpu()

    print(f"\nExport complete: {engine_root}")
    if not args.export_backbone:
        print("  Note: mot_backbone/ was not exported. C++ denoise requires it unless you add an eager bridge.")
    print("  Note: current C++ EmbedRunner expects 2 inputs; Omni embed engine has 3 (vision+sound+timestep).")
    print("  Note: use --dtype fp16 for compatibility with current C++ runner dtype parsing.")
    return engine_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Cosmos3-Omni WFM TRT engines for Edge-LLM runtime.")
    parser.add_argument("--engine-dir", required=True, help="Output directory for the engine bundle.")
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--prompt", default="A robot arm picks up a red cube while a motor whirs.")
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Tensor dtype for export modules and config metadata (fp16 matches current C++ runners).",
    )
    parser.add_argument(
        "--export-backbone",
        action="store_true",
        help="Also compile mot_backbone (Nano needs ~48GB GPU during export).",
    )
    args = parser.parse_args()
    export_engines(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
