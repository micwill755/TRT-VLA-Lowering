import sys
import os

import torch
import torch_tensorrt
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parents[1]

if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.compile import make_input_spec
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention, free_cuda_memory
from trt.modules.cosmos.packing import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from trt.modules.cosmos.vision import (
    Cosmos3MoTLayerExportModule,
)

MODEL_ID = "nvidia/Cosmos3-Nano"

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
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


def load_cosmos_components(dtype=torch.bfloat16):
    from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from transformers import Qwen2TokenizerFast

    transformer = Cosmos3OmniTransformer.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_ID, subfolder="text_tokenizer")
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    return {
        "transformer": transformer,
        "vae": vae,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    load_plugins_for_trt()

    components = load_cosmos_components(dtype=dtype)
    transformer = components["transformer"]

    force_hf_attention(transformer, "eager")

    und_seq, gen_seq, rotary_emb = make_layer_sample_inputs(
        transformer,
        device=device,
        dtype=dtype,
    )

    # Keep only one decoder layer on the GPU. The full transformer remains on
    # CPU, avoiding the full 16B graph that OOMs during TRT lowering.
    layer = transformer.layers[0].to(device=device).eval()

    components["transformer"] = None
    components["vae"] = None
    transformer = None
    free_cuda_memory()

    layer_module = Cosmos3MoTLayerExportModule(
        layer,
        sample_und_seq=und_seq,
        sample_gen_seq=gen_seq,
        sample_rotary_emb=rotary_emb,
    ).eval()

    with torch.no_grad():
        und_eager, gen_eager = layer_module(und_seq, gen_seq, *rotary_emb)

    exported = None
    trt_engine = None
    
    try:
        exported = torch.export.export(
            layer_module,
            args=(und_seq, gen_seq, *rotary_emb),
            strict=False,
        )
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=make_input_spec((und_seq, gen_seq, *rotary_emb)),
            **TRT_SETTINGS,
        )
        with torch.no_grad():
            und_trt, gen_trt = trt_engine(und_seq, gen_seq, *rotary_emb)
    finally:
        free_cuda_memory(trt_engine, exported, layer_module, layer)

    return 0


if __name__ == "__main__":
    SystemExit(main())
