"""Cosmos3-Edge multi-engine e2e — built piece by piece in this script only.

Piece 1: golden eager denoise step
Piece 2: visual_encode TRT (Wan VAE)
Piece 3: embed TRT (patchify + proj_in + timestep -> gen_seq vision)
Piece 4: mot_backbone TRT (und_seq + gen_seq + rotary -> last_hidden_state)
Piece 5: denoise_head TRT (last_hidden -> pred vision latents) + staged chain vs golden
Piece 6: visual_decode TRT (pred latents -> pixels)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch_tensorrt
from accelerate.utils import set_module_tensor_to_device
from diffusers import Cosmos3OmniPipeline

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.compile import make_input_spec
from trt.measure import parity
from trt.modules.cosmos.backbone import Cosmos3MoTBackboneExportModule
from trt.modules.cosmos.head import Cosmos3VisionDenoiseHeadExportModule
from trt.modules.cosmos.embed import Cosmos3VisionGenEmbedExportModule
from trt.modules.cosmos.packing import build_cosmos_packed_static, decode_cosmos_video, encode_cosmos_video
from trt.modules.cosmos.decode import CosmosVaeDecodeExportModule
from trt.modules.cosmos.vision import CosmosVaeEncodeExportModule
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention

MODEL_ID = "nvidia/Cosmos3-Edge"

# Small shapes for fast iteration; scale up after parity.
NUM_FRAMES = 9
HEIGHT = 256
WIDTH = 256
PROMPT = "A robot arm picks up a red cube."
TIMESTEP = 0.5
SEED = 42

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}


def fix_edge_diffusers_weights(transformer, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> None:
    """Materialize Edge checkpoint tensors that diffusers leaves on meta device."""
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


def load_cosmos_from_pipeline(*, dtype: torch.dtype = torch.bfloat16, device: torch.device | str = "cuda") -> dict:
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
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
    """CPU text embed -> und_seq [und_len, hidden]."""
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


@torch.no_grad()
def eager_vision_denoise_step(
    transformer,
    packed_static: dict,
    vision_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    """Golden reference for one denoise step (vision prediction only)."""
    kwargs = build_vision_denoise_kwargs(packed_static, vision_latents, timestep)
    pred_vision, _, _ = transformer(**kwargs)
    return pred_vision[0]


@torch.no_grad()
def eager_vision_gen_embed(
    transformer,
    packed_static: dict,
    vision_latents: torch.Tensor,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    """Eager vision gen_seq: patchify + proj_in + timestep embed -> [num_tokens, hidden]."""
    packed_tokens_vision, _ = transformer._patchify_and_pack_latents([vision_latents])
    packed_tokens_vision = transformer.proj_in(packed_tokens_vision)

    num_noisy = int(packed_static["num_noisy_tokens"])
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=vision_latents.device, dtype=vision_latents.dtype)
    timestep = timestep.to(device=vision_latents.device, dtype=vision_latents.dtype).reshape(())
    vision_timesteps = timestep.expand(num_noisy) * transformer.config.timestep_scale
    packed_timestep_embeds = transformer.time_embedder(transformer.time_proj(vision_timesteps))
    packed_timestep_embeds = packed_timestep_embeds.to(dtype=packed_tokens_vision.dtype)
    return transformer._apply_timestep_embeds_to_noisy_tokens(
        packed_tokens=packed_tokens_vision,
        packed_timestep_embeds=packed_timestep_embeds,
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
    """Eager MoT backbone: layers + norm -> last_hidden_state."""
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
    """Eager vision head: proj_out + unpatchify -> pred latents."""
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
def seed_noisy_vision_latents(
    latents: torch.Tensor,
    noisy_frame_indexes: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Keep conditioning frames clean; noise the generation frames."""
    out = latents.clone()
    noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
    for frame_idx in noisy_frame_indexes.tolist():
        out[:, :, frame_idx] = noise[:, :, frame_idx]
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


def _compile_trt(module: torch.nn.Module, sample_inputs: tuple) -> torch.nn.Module:
    exported = torch.export.export(module, args=sample_inputs, strict=False)
    input_specs = make_input_spec(sample_inputs)
    return torch_tensorrt.dynamo.compile(
        exported,
        inputs=input_specs,
        **{**TRT_SETTINGS, "use_python_runtime": True},
    )


def main() -> int:
    # Shape flow for this config (NUM_FRAMES=9, HEIGHT=WIDTH=256, hidden=2048):
    #   pixels          [B, 3, T, H, W]           = [1, 3, 9, 256, 256]
    #   latents         [B, C_lat, T', H', W']     = [1, 48, 3, 16, 16]
    #   und_seq         [und_len, hidden]          = [74, 2048]
    #   gen_seq         [num_vision_tokens, hidden]= [192, 2048]   (3*8*8 patches)
    #   last_hidden     [sequence_length, hidden]  = [266, 2048]   (74 + 192)
    #   pred_latents    [B, C_lat, T', H', W']     = [1, 48, 3, 16, 16]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    load_plugins_for_trt()

    print("[1] Load Cosmos3-Edge")
    components = load_cosmos_from_pipeline(dtype=dtype, device=device)
    transformer = components["transformer"]
    vae = components["vae"]
    tokenizer = components["tokenizer"]
    force_hf_attention(transformer, "eager")

    print("[2] CPU packing (Phase 0)")
    # CPU: tokenize + mRoPE + VAE encode -> static indexes + clean latents
    packed_static, clean_latents, pixels = build_cosmos_packed_static(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        device=device,
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=24.0,
        condition_frame_indexes=(0,),
        dtype=dtype,
    )
    und_len = int(packed_static["und_len"])
    # packed_static["input_ids"]           [und_len]              = [74]
    # packed_static["position_ids"]        [3, sequence_length]   = [3, 266]
    # packed_static["vision_token_shapes"] [(T', patch_h, patch_w)] = [(3, 8, 8)]
    # packed_static["vision_mse_loss_indexes"] [num_noisy_tokens] = [128]  (2 noisy frames * 64)
    print(f"  und_len={und_len} sequence_length={packed_static['sequence_length']}")
    print(f"  vision_token_shapes={packed_static['vision_token_shapes']}")
    print(f"  num_noisy_tokens={packed_static['num_noisy_tokens']}")
    _print_shape("pixels", pixels)           # [1, 3, 9, 256, 256]
    _print_shape("clean_latents", clean_latents)  # [1, 48, 3, 16, 16]

    print("\n[3] CPU text embed -> und_seq")
    und_seq = build_und_seq(transformer, packed_static["input_ids"])
    _print_shape("und_seq", und_seq)  # [und_len, hidden] = [74, 2048]

    print("\n[4] CPU rotary from position_ids")
    rotary_emb = build_rotary_emb(
        transformer,
        packed_static["position_ids"],  # [3, sequence_length]
        und_len,
        device=device,
        dtype=dtype,
    )
    # rotary_emb: (cos_und, sin_und, cos_gen, sin_gen)
    #   cos_und/sin_und [und_len, head_dim]
    #   cos_gen/sin_gen [num_vision_tokens, head_dim] = [192, head_dim]
    for name, tensor in zip(("cos_und", "sin_und", "cos_gen", "sin_gen"), rotary_emb):
        _print_shape(name, tensor)

    print("\n[5] Seed noisy vision latents")
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    noisy_latents = seed_noisy_vision_latents(
        clean_latents,  # [1, 48, 3, 16, 16] — frame 0 kept clean, frames 1-2 noised
        packed_static["vision_noisy_frame_indexes"][0],
        generator=generator,
    )
    _print_shape("noisy_latents", noisy_latents)  # [1, 48, 3, 16, 16]

    print("\n=== Piece 1: golden eager denoise step ===")
    # Full transformer: embed + 28 MoT layers + head (golden reference)
    pred_latents = eager_vision_denoise_step(
        transformer,
        packed_static,
        noisy_latents,  # IN:  [1, 48, 3, 16, 16]
        TIMESTEP,       # scalar t
    )
    _print_shape("pred_vision_latents", pred_latents)  # OUT: [1, 48, 3, 16, 16]

    print("\n=== Piece 2: visual_encode TRT ===")
    # Engine 1: pixels -> normalized VAE latents
    visual_encode = CosmosVaeEncodeExportModule(vae, pixels).eval().to(device)
    encode_inputs = (pixels,)  # [1, 3, 9, 256, 256]
    with torch.no_grad():
        latents_eager = visual_encode(*encode_inputs)  # OUT: [1, 48, 3, 16, 16]
        latents_ref = encode_cosmos_video(vae, pixels)
    parity("visual_encode module vs packing encode", latents_eager, latents_ref)

    trt_visual_encode = _compile_trt(visual_encode, encode_inputs)
    with torch.no_grad():
        latents_trt = trt_visual_encode(*encode_inputs)  # OUT: [1, 48, 3, 16, 16]
    parity("visual_encode eager vs TRT", latents_eager, latents_trt)
    encode_eager_ms = _time_ms(lambda: visual_encode(*encode_inputs), device=device)
    encode_trt_ms = _time_ms(lambda: trt_visual_encode(*encode_inputs), device=device)
    print(f"  visual_encode eager: {encode_eager_ms:.3f} ms")
    print(f"  visual_encode trt:   {encode_trt_ms:.3f} ms")
    print(f"  visual_encode speedup: {encode_eager_ms / encode_trt_ms:.3f}x")

    print("\n=== Piece 3: embed -> gen_seq vision ===")
    # Engine 2: patchify + proj_in + timestep embed -> gen_seq (vision tokens only)
    timestep_t = torch.tensor(TIMESTEP, device=device, dtype=dtype)  # scalar
    gen_seq_vision = eager_vision_gen_embed(
        transformer,
        packed_static,
        noisy_latents,  # IN:  [1, 48, 3, 16, 16]
        timestep_t,
    )
    _print_shape("gen_seq_vision (eager)", gen_seq_vision)  # OUT: [192, 2048]

    embed_module = Cosmos3VisionGenEmbedExportModule(
        transformer,
        packed_static=packed_static,
        sample_latents=noisy_latents,
        sample_timestep=TIMESTEP,
    ).eval().to(device)
    embed_inputs = (noisy_latents, timestep_t)  # ([1,48,3,16,16], scalar)
    with torch.no_grad():
        gen_seq_module = embed_module(*embed_inputs)  # OUT: [192, 2048]
    parity("embed module vs eager", gen_seq_module, gen_seq_vision)

    trt_embed = _compile_trt(embed_module, embed_inputs)
    with torch.no_grad():
        gen_seq_trt = trt_embed(*embed_inputs)  # OUT: [192, 2048]
    parity("embed eager vs TRT", gen_seq_module, gen_seq_trt)
    embed_eager_ms = _time_ms(lambda: embed_module(*embed_inputs), device=device)
    embed_trt_ms = _time_ms(lambda: trt_embed(*embed_inputs), device=device)
    print(f"  embed eager: {embed_eager_ms:.3f} ms")
    print(f"  embed trt:   {embed_trt_ms:.3f} ms")
    print(f"  embed speedup: {embed_eager_ms / embed_trt_ms:.3f}x")

    print("\n=== Piece 4: mot_backbone ===")
    # Engine 3: 28 MoT layers + norm — und_seq and gen_seq stay separate until concat
    gen_seq = gen_seq_vision  # Edge: gen_seq is vision-only [192, 2048]
    _print_shape("gen_seq (vision slice)", gen_seq)

    last_hidden_eager = eager_mot_backbone(
        transformer,
        und_seq,      # IN: [74, 2048]
        gen_seq,      # IN: [192, 2048]
        rotary_emb,   # IN: cos/sin split by und vs gen
    )
    _print_shape("last_hidden_state (eager)", last_hidden_eager)  # OUT: [266, 2048]

    backbone_module = Cosmos3MoTBackboneExportModule(
        transformer,
        sample_und_seq=und_seq,
        sample_gen_seq=gen_seq,
        sample_rotary_emb=rotary_emb,
    ).eval().to(device)
    backbone_inputs = (und_seq, gen_seq, *rotary_emb)
    # IN: und_seq [74,2048], gen_seq [192,2048], cos_und/sin_und [74,*], cos_gen/sin_gen [192,*]
    with torch.no_grad():
        last_hidden_module = backbone_module(*backbone_inputs)  # OUT: [266, 2048]
    parity("mot_backbone module vs eager", last_hidden_module, last_hidden_eager)

    print("  compiling mot_backbone (28 layers, may take several minutes)...")
    trt_backbone = _compile_trt(backbone_module, backbone_inputs)
    with torch.no_grad():
        last_hidden_trt = trt_backbone(*backbone_inputs)  # OUT: [266, 2048]
    parity("mot_backbone eager vs TRT", last_hidden_module, last_hidden_trt)
    backbone_eager_ms = _time_ms(
        lambda: backbone_module(*backbone_inputs),
        device=device,
        iters=20,
    )
    backbone_trt_ms = _time_ms(
        lambda: trt_backbone(*backbone_inputs),
        device=device,
        iters=20,
    )
    print(f"  mot_backbone eager: {backbone_eager_ms:.3f} ms")
    print(f"  mot_backbone trt:   {backbone_trt_ms:.3f} ms")
    print(f"  mot_backbone speedup: {backbone_eager_ms / backbone_trt_ms:.3f}x")

    print("\n=== Piece 5: denoise_head ===")
    # Engine 4: proj_out on noisy vision token rows -> unpatchify -> pred latents
    pred_head_eager = eager_vision_denoise_head(
        transformer,
        packed_static,
        last_hidden_eager,  # IN: [266, 2048] — only vision_mse_loss_indexes rows used
        noisy_latents,      # for unpatchify shape reference
    )
    _print_shape("pred_vision_latents (head eager)", pred_head_eager)  # OUT: [1, 48, 3, 16, 16]
    parity("denoise_head vs piece1 golden", pred_head_eager, pred_latents)

    head_module = Cosmos3VisionDenoiseHeadExportModule(
        transformer,
        packed_static=packed_static,
        sample_latents=noisy_latents,
        sample_last_hidden=last_hidden_eager,
    ).eval().to(device)
    head_inputs = (last_hidden_eager,)  # [266, 2048]
    with torch.no_grad():
        pred_head_module = head_module(*head_inputs)  # OUT: [1, 48, 3, 16, 16]
    parity("denoise_head module vs eager", pred_head_module, pred_head_eager)

    trt_head = _compile_trt(head_module, head_inputs)
    with torch.no_grad():
        pred_head_trt = trt_head(*head_inputs)  # OUT: [1, 48, 3, 16, 16]
    parity("denoise_head eager vs TRT", pred_head_module, pred_head_trt)
    head_eager_ms = _time_ms(lambda: head_module(*head_inputs), device=device)
    head_trt_ms = _time_ms(lambda: trt_head(*head_inputs), device=device)
    print(f"  denoise_head eager: {head_eager_ms:.3f} ms")
    print(f"  denoise_head trt:   {head_trt_ms:.3f} ms")
    print(f"  denoise_head speedup: {head_eager_ms / head_trt_ms:.3f}x")

    print("\n=== Staged chain: embed -> backbone -> head (eager) ===")
    # One denoise step decomposed: gen_seq [192,2048] -> last_hidden [266,2048] -> pred [1,48,3,16,16]
    with torch.no_grad():
        chain_hidden = eager_mot_backbone(
            transformer,
            und_seq,
            eager_vision_gen_embed(transformer, packed_static, noisy_latents, timestep_t),
            rotary_emb,
        )
        pred_chain_eager = eager_vision_denoise_head(
            transformer,
            packed_static,
            chain_hidden,
            noisy_latents,
        )
    parity("staged eager chain vs piece1 golden", pred_chain_eager, pred_latents)

    print("\n=== Staged chain: embed -> backbone -> head (TRT) ===")
    with torch.no_grad():
        chain_gen_trt = trt_embed(*embed_inputs)           # [192, 2048]
        chain_hidden_trt = trt_backbone(und_seq, chain_gen_trt, *rotary_emb)  # [266, 2048]
        pred_chain_trt = trt_head(chain_hidden_trt)        # [1, 48, 3, 16, 16]
    parity("staged TRT chain vs piece1 golden", pred_chain_trt, pred_latents)

    print("\n=== Piece 6: visual_decode ===")
    # Engine 5: pred latents -> pixels (run once after scheduler loop)
    decode_inputs = (pred_latents,)  # [1, 48, 3, 16, 16]
    pixels_decoded = decode_cosmos_video(vae, pred_latents)
    _print_shape("decoded_pixels (eager)", pixels_decoded)  # OUT: [1, 3, 9, 256, 256]

    decode_module = CosmosVaeDecodeExportModule(vae, pred_latents).eval().to(device)
    with torch.no_grad():
        pixels_module = decode_module(*decode_inputs)  # OUT: [1, 3, 9, 256, 256]
    parity("visual_decode module vs eager", pixels_module, pixels_decoded)

    print("  compiling visual_decode...")
    trt_decode = _compile_trt(decode_module, decode_inputs)
    with torch.no_grad():
        pixels_trt = trt_decode(*decode_inputs)  # OUT: [1, 3, 9, 256, 256]
    parity("visual_decode eager vs TRT", pixels_module, pixels_trt)
    decode_eager_ms = _time_ms(lambda: decode_module(*decode_inputs), device=device, iters=20)
    decode_trt_ms = _time_ms(lambda: trt_decode(*decode_inputs), device=device, iters=20)
    print(f"  visual_decode eager: {decode_eager_ms:.3f} ms")
    print(f"  visual_decode trt:   {decode_trt_ms:.3f} ms")
    print(f"  visual_decode speedup: {decode_eager_ms / decode_trt_ms:.3f}x")

    print("\nAll 6 pieces complete — full Edge denoise pipeline validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
