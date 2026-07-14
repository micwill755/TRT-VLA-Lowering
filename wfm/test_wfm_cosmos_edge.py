from __future__ import annotations

import logging
import sys
import time
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
from trt.modules.cosmos.packing import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from trt.modules.cosmos.vision import Cosmos3MoTLayerExportModule
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention

MODEL_ID = "nvidia/Cosmos3-Edge"

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
}


def fix_edge_diffusers_weights(transformer, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> None:
    """Materialize Edge checkpoint tensors that diffusers leaves on meta device.

    The Edge shard stores SwiGLU ``up_proj``/``down_proj`` but not ``gate_proj``, and
    omits per-head Q/K RMSNorm weights because ``qk_norm_for_text=False``.
    """
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
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    fix_edge_diffusers_weights(pipe.transformer, device="cpu", dtype=dtype)

    return {
        "transformer": pipe.transformer.to(device).eval(),
        "vae": pipe.vae.to(device).eval(),
        "tokenizer": pipe.text_tokenizer,
        "scheduler": pipe.scheduler,
    }


def make_layer_sample_inputs(
    transformer,
    *,
    device: torch.device,
    dtype: torch.dtype,
    und_len: int = 128,
    latent_t: int = 5,
    patch_h: int = 15,
    patch_w: int = 15,
    fps: float = 24.0,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    cfg = transformer.config
    hidden_size = int(cfg.hidden_size)

    text_mrope_ids, next_offset = get_3d_mrope_ids_text_tokens(
        und_len,
        temporal_offset=0,
        use_float_positions=cfg.enable_fps_modulation,
    )
    vision_start_offset = next_offset + cfg.unified_3d_mrope_temporal_modality_margin
    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=vision_start_offset,
        reset_spatial_indices=cfg.unified_3d_mrope_reset_spatial_ids,
        fps=fps if cfg.enable_fps_modulation else None,
        base_fps=float(cfg.base_fps),
        temporal_compression_factor=4,
    )
    position_ids = torch.cat([text_mrope_ids, vision_mrope_ids], dim=-1).to(device)

    gen_len = int(vision_mrope_ids.shape[-1])
    und_seq = torch.randn(und_len, hidden_size, device=device, dtype=dtype)
    gen_seq = torch.randn(gen_len, hidden_size, device=device, dtype=dtype)

    cos, sin = transformer.rotary_emb(
        position_ids=position_ids.unsqueeze(1),
        device=device,
        dtype=dtype,
    )
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    rotary_emb = (
        cos[:und_len].contiguous(),
        sin[:und_len].contiguous(),
        cos[und_len:].contiguous(),
        sin[und_len:].contiguous(),
    )
    return und_seq, gen_seq, rotary_emb


def _time_ms(fn, *, warmup: int = 5, iters: int = 100, device: torch.device) -> float:
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Cosmos vision TRT export.")

    dtype = torch.bfloat16
    load_plugins_for_trt()

    components = load_cosmos_from_pipeline(dtype=dtype, device=device)
    transformer = components["transformer"]
    force_hf_attention(transformer, "eager")

    # STEP 1 vision
    print("Compiling vision")

    und_seq, gen_seq, rotary_emb = make_layer_sample_inputs(
        transformer,
        device=device,
        dtype=dtype,
    )
    sample_inputs = (und_seq, gen_seq, *rotary_emb)

    # The Cosmos3 dual-path MoT decoder layer is the vision compute block and the
    # practical TRT export boundary (the full 28-layer transformer OOMs the builder).
    layer = transformer.layers[0].to(device=device).eval()
    visual = Cosmos3MoTLayerExportModule(
        layer,
        sample_und_seq=und_seq,
        sample_gen_seq=gen_seq,
        sample_rotary_emb=rotary_emb,
    ).eval().to(device=device)

    # --- Rung A: eager ---
    with torch.no_grad():
        und_eager, gen_eager = visual(*sample_inputs)

    vision_eager_elapsed_ms = _time_ms(
        lambda: visual(*sample_inputs),
        device=device,
    )

    # --- Rung C: TRT compiled ---
    exported = torch.export.export(visual, args=sample_inputs, strict=False)
    input_specs = make_input_spec(sample_inputs)
    trt_engine = torch_tensorrt.dynamo.compile(
        exported,
        inputs=input_specs,
        **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
    )
    with torch.no_grad():
        und_trt, gen_trt = trt_engine(*sample_inputs)

    vision_trt_elapsed_ms = _time_ms(
        lambda: trt_engine(*sample_inputs),
        device=device,
    )

    parity("vision und A vs C (prod)", und_eager, und_trt)
    parity("vision gen A vs C (prod)", gen_eager, gen_trt)

    print(f"vision eager execute: {vision_eager_elapsed_ms:.3f} ms")
    print(f"vision trt execute: {vision_trt_elapsed_ms:.3f} ms")
    print(f"vision speedup: {(vision_eager_elapsed_ms / vision_trt_elapsed_ms):.3f}x")
    return 0


if __name__ == "__main__":
    SystemExit(main())
