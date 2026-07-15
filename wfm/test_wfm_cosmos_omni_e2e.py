"""Cosmos3-Omni multi-engine e2e — mirrors ``test_wfm_cosmos_edge_e2e.py`` with sound.

Piece 1: golden eager denoise step (vision + sound)
Piece 2: visual_encode TRT (Wan VAE)
Piece 2b: audio_encode TRT (AVAE full waveform->latent; conv-STFT replaces FFT)
Piece 3: embed TRT (vision + sound -> fused gen_seq)
Piece 4: mot_backbone TRT
Piece 5: denoise_head TRT (vision)
Piece 5b: denoise_head_sound TRT
Piece 6: visual_decode TRT
Piece 6b: audio_decode TRT
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch_tensorrt
from diffusers import Cosmos3OmniPipeline

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.compile import make_input_spec
from trt.measure import parity
from trt.modules.cosmos.audio import (
    CosmosAvaeDecodeExportModule,
    CosmosAvaeEncodeConvStftExportModule,
    CosmosAvaeEncodeExportModule,
)
from trt.modules.cosmos.backbone import Cosmos3MoTBackboneExportModule
from trt.modules.cosmos.decode import CosmosVaeDecodeExportModule
from trt.modules.cosmos.embed import Cosmos3OmniGenEmbedExportModule
from trt.modules.cosmos.head import Cosmos3SoundDenoiseHeadExportModule, Cosmos3VisionDenoiseHeadExportModule
from trt.modules.cosmos.packing import (
    build_cosmos_omni_packed_static,
    decode_cosmos_sound,
    decode_cosmos_video,
    encode_cosmos_video,
)
from trt.modules.cosmos.vision import CosmosVaeEncodeExportModule
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention

MODEL_ID = "nvidia/Cosmos3-Nano"

# Small shapes for fast iteration; Nano backbone compile is still heavy (36 layers).
NUM_FRAMES = 9
HEIGHT = 256
WIDTH = 256
PROMPT = "A robot arm picks up a red cube while a motor whirs."
TIMESTEP = 0.5
SEED = 42

# Nano's MoT backbone is ~16B params (~31GB bf16); a single TRT engine for the full
# 36-layer backbone does not fit alongside the resident eager transformer on a 32GB GPU.
# The backbone is architecturally identical to Edge (validated there); default it off and
# keep eager parity. Set COMPILE_BACKBONE_TRT=1 to force it (needs a larger GPU).
COMPILE_BACKBONE_TRT = os.environ.get("COMPILE_BACKBONE_TRT", "0") == "1"

# AVAE Oobleck decoder uses bf16 ConvTranspose1d (deconv) + weight_norm + Snake1d; TensorRT builder
# fails to find tactics for this graph on our stack. Eager parity validates the export module.
COMPILE_AUDIO_DECODE_TRT = os.environ.get("COMPILE_AUDIO_DECODE_TRT", "0") == "1"

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
}


# Nano (~16B) fills a 32GB GPU; sound_tokenizer is staged on GPU only for AVAE TRT.


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def stage_transformer_vae_on_gpu(transformer, vae, device: torch.device) -> None:
    """Move the large transformer (+ VAE) to GPU; keep sound_tokenizer on CPU."""
    transformer.to(device)
    vae.to(device)
    _sync_gpu()


def stage_vae_only_on_gpu(transformer, vae, device: torch.device) -> None:
    """VAE TRT compile/decode — transformer must leave GPU on 32GB cards."""
    transformer.to("cpu")
    vae.to(device)
    _sync_gpu()


def stage_transformer_only_on_gpu(transformer, vae, device: torch.device) -> None:
    """Transformer TRT compile (embed / backbone / heads)."""
    vae.to("cpu")
    transformer.to(device)
    _sync_gpu()


def stage_sound_tokenizer_on_gpu(transformer, vae, sound_tokenizer, device: torch.device) -> None:
    """Free transformer/VAE VRAM so AVAE encode/decode TRT can run on a 32GB GPU."""
    transformer.to("cpu")
    vae.to("cpu")
    _sync_gpu()
    sound_tokenizer.to(device)
    _sync_gpu()


def release_sound_tokenizer_from_gpu(sound_tokenizer) -> None:
    sound_tokenizer.to("cpu")
    _sync_gpu()


def load_cosmos_from_pipeline(*, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Load pipeline on CPU; caller stages components onto GPU as needed."""
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        enable_safety_checker=False,
    )
    if pipe.sound_tokenizer is None:
        raise RuntimeError(f"{MODEL_ID} is missing sound_tokenizer; use an Omni-capable checkpoint.")
    if not getattr(pipe.transformer.config, "sound_gen", False):
        raise RuntimeError(f"{MODEL_ID} transformer.config.sound_gen=False.")
    return {
        "transformer": pipe.transformer.eval(),
        "vae": pipe.vae.eval(),
        "sound_tokenizer": pipe.sound_tokenizer.eval(),
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


def build_vision_denoise_kwargs(
    packed_static: dict,
    vision_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> dict:
    num_noisy = int(packed_static["num_noisy_tokens"])
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=vision_latents.device, dtype=vision_latents.dtype)
    timestep = timestep.to(device=vision_latents.device, dtype=vision_latents.dtype).reshape(())
    vision_timesteps = timestep.expand(num_noisy)
    return dict(
        input_ids=packed_static["input_ids"],
        text_indexes=packed_static["text_indexes"],
        position_ids=packed_static["position_ids"],
        und_len=int(packed_static["und_len"]),
        sequence_length=int(packed_static["sequence_length"]),
        vision_tokens=[vision_latents],
        vision_token_shapes=packed_static["vision_token_shapes"],
        vision_sequence_indexes=packed_static["vision_sequence_indexes"],
        vision_mse_loss_indexes=packed_static["vision_mse_loss_indexes"],
        vision_timesteps=vision_timesteps,
        vision_noisy_frame_indexes=packed_static["vision_noisy_frame_indexes"],
    )


def build_omni_denoise_kwargs(
    packed_static: dict,
    vision_latents: torch.Tensor,
    sound_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> dict:
    kwargs = build_vision_denoise_kwargs(packed_static, vision_latents, timestep)
    sound_len = int(packed_static["sound_len"])
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=vision_latents.device, dtype=vision_latents.dtype)
    timestep = timestep.to(device=vision_latents.device, dtype=vision_latents.dtype).reshape(())
    sound_timesteps = timestep.expand(sound_len)
    kwargs.update(
        sound_tokens=[sound_latents],
        sound_token_shapes=packed_static["sound_token_shapes"],
        sound_sequence_indexes=packed_static["sound_sequence_indexes"],
        sound_mse_loss_indexes=packed_static["sound_mse_loss_indexes"],
        sound_timesteps=sound_timesteps,
        sound_noisy_frame_indexes=packed_static["sound_noisy_frame_indexes"],
    )
    return kwargs


@torch.no_grad()
def eager_omni_denoise_step(
    transformer,
    packed_static: dict,
    vision_latents: torch.Tensor,
    sound_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = build_omni_denoise_kwargs(packed_static, vision_latents, sound_latents, timestep)
    pred_vision, pred_sound, _ = transformer(**kwargs)
    return pred_vision[0], pred_sound[0]


@torch.no_grad()
def eager_omni_gen_embed(
    transformer,
    packed_static: dict,
    vision_latents: torch.Tensor,
    sound_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    module = Cosmos3OmniGenEmbedExportModule(
        transformer,
        packed_static=packed_static,
        sample_vision_latents=vision_latents,
        sample_sound_latents=sound_latents,
        sample_timestep=float(timestep) if not torch.is_tensor(timestep) else float(timestep.item()),
    ).eval().to(device=vision_latents.device)
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=vision_latents.device, dtype=vision_latents.dtype)
    return module(vision_latents, sound_latents, timestep)


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
    und_out = transformer.norm(und_seq)
    gen_out = transformer.norm_moe_gen(gen_seq)
    return torch.cat([und_out, gen_out], dim=0)


@torch.no_grad()
def eager_vision_denoise_head(
    transformer,
    packed_static: dict,
    last_hidden_state: torch.Tensor,
    vision_latents: torch.Tensor,
) -> torch.Tensor:
    preds_packed = transformer.proj_out(last_hidden_state[packed_static["vision_mse_loss_indexes"]])
    _, original_latent_shapes = transformer._patchify_and_pack_latents([vision_latents])
    preds = transformer._unpatchify_and_unpack_latents(
        preds_packed,
        token_shapes_vision=packed_static["vision_token_shapes"],
        noisy_frame_indexes_vision=packed_static["vision_noisy_frame_indexes"],
        original_latent_shapes=original_latent_shapes,
    )
    return preds[0]


@torch.no_grad()
def eager_sound_denoise_head(
    transformer,
    packed_static: dict,
    last_hidden_state: torch.Tensor,
) -> torch.Tensor:
    preds_packed = transformer.audio_proj_out(last_hidden_state[packed_static["sound_mse_loss_indexes"]])
    preds = transformer._unpack_sound_latents(
        preds_packed,
        packed_static["sound_token_shapes"],
        packed_static["sound_noisy_frame_indexes"],
    )
    return preds[0]


@torch.no_grad()
def seed_noisy_vision_latents(
    latents: torch.Tensor,
    noisy_frame_indexes: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    out = latents.clone()
    noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
    for frame_idx in noisy_frame_indexes.tolist():
        out[:, :, frame_idx] = noise[:, :, frame_idx]
    return out


@torch.no_grad()
def seed_noisy_sound_latents(
    latents: torch.Tensor,
    noisy_slot_indexes: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    out = latents.clone()
    noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
    for slot_idx in noisy_slot_indexes.tolist():
        out[:, slot_idx] = noise[:, slot_idx]
    return out


def _print_shape(name: str, tensor: torch.Tensor) -> None:
    print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")


def _time_ms(fn, *, warmup: int = 5, iters: int = 50, device: torch.device) -> float:
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


def _compile_trt(
    module: torch.nn.Module,
    sample_inputs: tuple,
    *,
    device: torch.device | None = None,
    trt_overrides: dict | None = None,
) -> torch.nn.Module:
    _sync_gpu()
    exported = torch.export.export(module, args=sample_inputs, strict=False)
    input_specs = make_input_spec(sample_inputs)
    compile_settings = {**TRT_SETTINGS, "use_python_runtime": True, **(trt_overrides or {})}
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        inputs=input_specs,
        **compile_settings,
    )
    _sync_gpu()
    # offload_module_to_cpu leaves the source module's weights on CPU; the wrapped
    # vae/transformer params are shared, so re-stage them for eager parity/timing.
    if device is not None:
        module.to(device)
        _sync_gpu()
    return compiled


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Cosmos Omni TRT export.")

    dtype = torch.bfloat16
    load_plugins_for_trt()

    print(f"[1] Load {MODEL_ID} on CPU (stage to GPU per component)")
    components = load_cosmos_from_pipeline(dtype=dtype)
    transformer = components["transformer"]
    vae = components["vae"]
    sound_tokenizer = components["sound_tokenizer"]
    tokenizer = components["tokenizer"]
    force_hf_attention(transformer, "eager")

    print("[1b] Stage transformer + VAE on GPU (sound_tokenizer stays on CPU)")
    stage_transformer_vae_on_gpu(transformer, vae, device)

    print("[2] CPU packing (Phase 0, vision + sound)")
    packed_static, clean_latents, pixels, clean_sound, waveform = build_cosmos_omni_packed_static(
        transformer=transformer,
        vae=vae,
        sound_tokenizer=sound_tokenizer,
        tokenizer=tokenizer,
        device=device,
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=24.0,
        condition_frame_indexes=(0,),
        dtype=dtype,
        enable_sound=True,
    )
    und_len = int(packed_static["und_len"])
    sound_len = int(packed_static["sound_len"])
    print(
        f"  und_len={und_len} sequence_length={packed_static['sequence_length']} "
        f"sound_len={sound_len}"
    )
    _print_shape("pixels", pixels)
    _print_shape("clean_latents", clean_latents)
    _print_shape("waveform", waveform)
    _print_shape("clean_sound", clean_sound)

    print("\n[3] CPU text embed -> und_seq")
    und_seq = build_und_seq(transformer, packed_static["input_ids"])
    _print_shape("und_seq", und_seq)

    print("\n[4] CPU rotary from position_ids")
    rotary_emb = build_rotary_emb(
        transformer,
        packed_static["position_ids"],
        und_len,
        device=device,
        dtype=dtype,
    )
    for name, tensor in zip(("cos_und", "sin_und", "cos_gen", "sin_gen"), rotary_emb):
        _print_shape(name, tensor)

    print("\n[5] Seed noisy vision + sound latents")
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    noisy_latents = seed_noisy_vision_latents(
        clean_latents,
        packed_static["vision_noisy_frame_indexes"][0],
        generator=generator,
    )
    generator.manual_seed(SEED + 1)
    noisy_sound = seed_noisy_sound_latents(
        clean_sound,
        packed_static["sound_noisy_frame_indexes"][0],
        generator=generator,
    )
    _print_shape("noisy_latents", noisy_latents)
    _print_shape("noisy_sound", noisy_sound)

    print("\n=== Piece 1: golden eager denoise step (vision + sound) ===")
    pred_vision, pred_sound = eager_omni_denoise_step(
        transformer,
        packed_static,
        noisy_latents,
        noisy_sound,
        TIMESTEP,
    )
    _print_shape("pred_vision_latents", pred_vision)
    _print_shape("pred_sound_latents", pred_sound)

    print("\n[mem] Offload transformer for VAE-only TRT (piece 2 / 6)")
    stage_vae_only_on_gpu(transformer, vae, device)

    print("\n=== Piece 2: visual_encode TRT ===")
    visual_encode = CosmosVaeEncodeExportModule(vae, pixels).eval().to(device)
    encode_inputs = (pixels,)
    with torch.no_grad():
        latents_eager = visual_encode(*encode_inputs)
        latents_ref = encode_cosmos_video(vae, pixels)
    parity("visual_encode module vs packing encode", latents_eager, latents_ref)
    trt_visual_encode = _compile_trt(visual_encode, encode_inputs, device=device)
    with torch.no_grad():
        latents_trt = trt_visual_encode(*encode_inputs)
    parity("visual_encode eager vs TRT", latents_eager, latents_trt)
    encode_eager_ms = _time_ms(lambda: visual_encode(*encode_inputs), device=device)
    encode_trt_ms = _time_ms(lambda: trt_visual_encode(*encode_inputs), device=device)
    print(f"  visual_encode speedup: {encode_eager_ms / encode_trt_ms:.3f}x")
    del visual_encode, trt_visual_encode
    _sync_gpu()

    print("\n[mem] Stage transformer for embed/backbone/head TRT (piece 3-5)")
    stage_transformer_only_on_gpu(transformer, vae, device)

    print("\n=== Piece 3: embed -> fused gen_seq (vision + sound) ===")
    timestep_t = torch.tensor(TIMESTEP, device=device, dtype=dtype)
    gen_seq_omni = eager_omni_gen_embed(
        transformer,
        packed_static,
        noisy_latents,
        noisy_sound,
        timestep_t,
    )
    _print_shape("gen_seq_omni (eager)", gen_seq_omni)

    embed_module = Cosmos3OmniGenEmbedExportModule(
        transformer,
        packed_static=packed_static,
        sample_vision_latents=noisy_latents,
        sample_sound_latents=noisy_sound,
        sample_timestep=TIMESTEP,
    ).eval().to(device)
    embed_inputs = (noisy_latents, noisy_sound, timestep_t)
    with torch.no_grad():
        gen_seq_module = embed_module(*embed_inputs)
    parity("omni embed module vs eager", gen_seq_module, gen_seq_omni)
    trt_embed = _compile_trt(embed_module, embed_inputs, device=device)
    with torch.no_grad():
        gen_seq_trt = trt_embed(*embed_inputs)
    parity("omni embed eager vs TRT", gen_seq_module, gen_seq_trt)
    embed_eager_ms = _time_ms(lambda: embed_module(*embed_inputs), device=device)
    embed_trt_ms = _time_ms(lambda: trt_embed(*embed_inputs), device=device)
    print(f"  omni embed speedup: {embed_eager_ms / embed_trt_ms:.3f}x")

    print("\n=== Piece 4: mot_backbone ===")
    gen_seq = gen_seq_omni
    last_hidden_eager = eager_mot_backbone(transformer, und_seq, gen_seq, rotary_emb)
    _print_shape("last_hidden_state (eager)", last_hidden_eager)

    backbone_module = Cosmos3MoTBackboneExportModule(
        transformer,
        sample_und_seq=und_seq,
        sample_gen_seq=gen_seq,
        sample_rotary_emb=rotary_emb,
    ).eval().to(device)
    backbone_inputs = (und_seq, gen_seq, *rotary_emb)
    with torch.no_grad():
        last_hidden_module = backbone_module(*backbone_inputs)
    parity("mot_backbone module vs eager", last_hidden_module, last_hidden_eager)

    trt_backbone = None
    if COMPILE_BACKBONE_TRT:
        print("  compiling mot_backbone (36 layers, may take several minutes)...")
        trt_backbone = _compile_trt(backbone_module, backbone_inputs, device=device)
        with torch.no_grad():
            last_hidden_trt = trt_backbone(*backbone_inputs)
        parity("mot_backbone eager vs TRT", last_hidden_module, last_hidden_trt)
        backbone_eager_ms = _time_ms(lambda: backbone_module(*backbone_inputs), device=device, iters=20)
        backbone_trt_ms = _time_ms(lambda: trt_backbone(*backbone_inputs), device=device, iters=20)
        print(f"  mot_backbone speedup: {backbone_eager_ms / backbone_trt_ms:.3f}x")
    else:
        print(
            "  [skip] mot_backbone TRT (Nano ~16B does not fit on a 32GB GPU alongside the eager "
            "transformer). Eager parity validated; set COMPILE_BACKBONE_TRT=1 to force on a larger GPU."
        )

    print("\n=== Piece 5: denoise_head (vision) ===")
    pred_head_vision = eager_vision_denoise_head(
        transformer,
        packed_static,
        last_hidden_eager,
        noisy_latents,
    )
    parity("vision denoise_head vs piece1 golden", pred_head_vision, pred_vision)

    vision_head_module = Cosmos3VisionDenoiseHeadExportModule(
        transformer,
        packed_static=packed_static,
        sample_latents=noisy_latents,
        sample_last_hidden=last_hidden_eager,
    ).eval().to(device)
    vision_head_inputs = (last_hidden_eager,)
    with torch.no_grad():
        pred_vision_module = vision_head_module(*vision_head_inputs)
    parity("vision denoise_head module vs eager", pred_vision_module, pred_head_vision)
    trt_vision_head = _compile_trt(vision_head_module, vision_head_inputs, device=device)
    with torch.no_grad():
        pred_vision_trt = trt_vision_head(*vision_head_inputs)
    parity("vision denoise_head eager vs TRT", pred_vision_module, pred_vision_trt)

    print("\n=== Piece 5b: denoise_head_sound ===")
    pred_head_sound = eager_sound_denoise_head(transformer, packed_static, last_hidden_eager)
    parity("sound denoise_head vs piece1 golden", pred_head_sound, pred_sound)

    sound_head_module = Cosmos3SoundDenoiseHeadExportModule(
        transformer,
        packed_static=packed_static,
        sample_last_hidden=last_hidden_eager,
    ).eval().to(device)
    sound_head_inputs = (last_hidden_eager,)
    with torch.no_grad():
        pred_sound_module = sound_head_module(*sound_head_inputs)
    parity("sound denoise_head module vs eager", pred_sound_module, pred_head_sound)
    trt_sound_head = _compile_trt(sound_head_module, sound_head_inputs, device=device)
    with torch.no_grad():
        pred_sound_trt = trt_sound_head(*sound_head_inputs)
    parity("sound denoise_head eager vs TRT", pred_sound_module, pred_sound_trt)

    print("\n=== Staged chain: embed -> backbone -> dual heads (eager) ===")
    with torch.no_grad():
        chain_gen = eager_omni_gen_embed(transformer, packed_static, noisy_latents, noisy_sound, timestep_t)
        chain_hidden = eager_mot_backbone(transformer, und_seq, chain_gen, rotary_emb)
        pred_chain_vision = eager_vision_denoise_head(transformer, packed_static, chain_hidden, noisy_latents)
        pred_chain_sound = eager_sound_denoise_head(transformer, packed_static, chain_hidden)
    parity("staged eager vision vs piece1", pred_chain_vision, pred_vision)
    parity("staged eager sound vs piece1", pred_chain_sound, pred_sound)

    print("\n=== Staged chain: embed -> backbone -> dual heads (TRT) ===")
    with torch.no_grad():
        chain_gen_trt = trt_embed(*embed_inputs)
        if trt_backbone is not None:
            chain_hidden_trt = trt_backbone(und_seq, chain_gen_trt, *rotary_emb)
            backbone_label = "TRT"
        else:
            # Bridge with the eager backbone so the TRT embed + TRT heads still compose end-to-end.
            chain_hidden_trt = eager_mot_backbone(transformer, und_seq, chain_gen_trt, rotary_emb)
            backbone_label = "eager-bridge"
        pred_chain_vision_trt = trt_vision_head(chain_hidden_trt)
        pred_chain_sound_trt = trt_sound_head(chain_hidden_trt)
    parity(f"staged TRT vision vs piece1 (backbone={backbone_label})", pred_chain_vision_trt, pred_vision)
    parity(f"staged TRT sound vs piece1 (backbone={backbone_label})", pred_chain_sound_trt, pred_sound)

    print("\n[mem] Offload transformer for VAE decode TRT (piece 6)")
    stage_vae_only_on_gpu(transformer, vae, device)

    print("\n=== Piece 6: visual_decode ===")
    decode_inputs = (pred_vision,)
    pixels_decoded = decode_cosmos_video(vae, pred_vision)
    decode_module = CosmosVaeDecodeExportModule(vae, pred_vision).eval().to(device)
    with torch.no_grad():
        pixels_module = decode_module(*decode_inputs)
    parity("visual_decode module vs eager", pixels_module, pixels_decoded)
    trt_decode = _compile_trt(decode_module, decode_inputs, device=device)
    with torch.no_grad():
        pixels_trt = trt_decode(*decode_inputs)
    parity("visual_decode eager vs TRT", pixels_module, pixels_trt)

    print("\n[swap] Offload transformer/VAE; stage sound_tokenizer on GPU for AVAE TRT")
    stage_sound_tokenizer_on_gpu(transformer, vae, sound_tokenizer, device)
    waveform_gpu = waveform.to(device=device, dtype=dtype)

    print("\n=== Piece 2b: audio_encode TRT (conv-STFT, full waveform->latent) ===")
    audio_encode = CosmosAvaeEncodeExportModule(sound_tokenizer, waveform_gpu).eval().to(device)
    audio_encode_inputs = (waveform_gpu,)
    with torch.no_grad():
        sound_eager = audio_encode(*audio_encode_inputs)[0]
    parity("audio_encode module vs packing encode", sound_eager, clean_sound)

    # torch.stft (FFT) does not lower in Torch-TensorRT. Reimplement the STFT front-end as a fixed
    # conv1d DFT bank (window-folded cos/sin kernels) so the whole waveform->latent path is a single
    # TRT engine. This is mathematically equivalent to the FFT front-end (validated below).
    conv_encode = CosmosAvaeEncodeConvStftExportModule(sound_tokenizer, waveform_gpu).eval().to(device)
    with torch.no_grad():
        sound_conv_eager = conv_encode(*audio_encode_inputs)[0]
    parity("audio_encode conv-STFT vs full eager (FFT)", sound_conv_eager, sound_eager)

    trt_audio_encode = _compile_trt(
        conv_encode,
        audio_encode_inputs,
        device=device,
        trt_overrides={"offload_module_to_cpu": False},
    )
    with torch.no_grad():
        sound_trt = trt_audio_encode(*audio_encode_inputs)[0]
    parity("audio_encode eager vs TRT", sound_conv_eager, sound_trt)
    audio_encode_eager_ms = _time_ms(lambda: conv_encode(*audio_encode_inputs), device=device)
    audio_encode_trt_ms = _time_ms(lambda: trt_audio_encode(*audio_encode_inputs), device=device)
    print(f"  audio_encode speedup: {audio_encode_eager_ms / audio_encode_trt_ms:.3f}x")

    print("\n=== Piece 6b: audio_decode ===")
    sound_decode_latents = pred_sound.unsqueeze(0)
    audio_decode_inputs = (sound_decode_latents,)
    waveform_decoded = decode_cosmos_sound(sound_tokenizer, pred_sound)
    audio_decode_module = CosmosAvaeDecodeExportModule(sound_tokenizer, sound_decode_latents).eval().to(device)
    with torch.no_grad():
        waveform_module = audio_decode_module(*audio_decode_inputs)[0]
    parity("audio_decode module vs eager", waveform_module, waveform_decoded)
    if COMPILE_AUDIO_DECODE_TRT:
        trt_audio_decode = _compile_trt(
            audio_decode_module,
            audio_decode_inputs,
            device=device,
            trt_overrides={"offload_module_to_cpu": False},
        )
        with torch.no_grad():
            waveform_trt = trt_audio_decode(*audio_decode_inputs)[0]
        parity("audio_decode eager vs TRT", waveform_module, waveform_trt)
    else:
        print(
            "  [skip] audio_decode TRT (Oobleck decoder bf16 deconv1d does not build on this TensorRT stack). "
            "Eager parity validated; set COMPILE_AUDIO_DECODE_TRT=1 to force."
        )

    release_sound_tokenizer_from_gpu(sound_tokenizer)

    print("\nAll Omni pieces complete — vision + sound denoise pipeline validated.")
    print(
        "  TRT: visual_encode/decode, omni embed, denoise heads (vision+sound), "
        "AVAE encode (conv-STFT, full waveform->latent)"
    )
    print(
        "  Eager: MoT backbone (32GB), AVAE Oobleck decoder (TRT builder limit)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
