import sys

import torch
import torch_tensorrt
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parents[1]

if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.compile import make_input_spec
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention, free_cuda_memory
from trt.modules.cosmos.packing import build_cosmos_packed_static
from trt.modules.cosmos.vision import (
    Cosmos3MoTDenoiseStepExportModule,
    CosmosVaeEncodeExportModule,
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
    vae = components["vae"]
    tokenizer = components["tokenizer"]

    force_hf_attention(transformer, "eager")

    pixels = torch.randn(1, 3, 17, 480, 480, device=device, dtype=dtype)

    vae = vae.to(device=device, dtype=dtype).eval()
    packed_static, latents, pixels = build_cosmos_packed_static(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        device=device,
        prompt="pick up the banana",
        height=480,
        width=480,
        num_frames=17,
        pixels=pixels,
        dtype=dtype,
    )

    # build_cosmos_packed_static already ran the standalone VAE encode. Keep the
    # latents needed by the MoT step, then drop VAE-side objects before loading
    # the full transformer onto the GPU for TRT export.
    latents_eager = latents.detach().contiguous()
    components["vae"] = None
    vae = None
    pixels = None
    latents = None
    free_cuda_memory()

    # Preserve module-specific dtypes from from_pretrained(). Cosmos keeps
    # time_embedder in fp32 while most weights are bf16; forcing dtype here
    # makes time_proj's fp32 output hit bf16 Linear weights.
    transformer = transformer.to(device=device).eval()
    timestep = torch.tensor(500.0, device=device)

    mot_step = Cosmos3MoTDenoiseStepExportModule(
        transformer=transformer,
        packed_static=packed_static,
        sample_latents=latents_eager,
        sample_timestep=500.0,
        modality="vision",
    ).eval()

    with torch.no_grad():
        vel_eager = mot_step(latents_eager, timestep)

    exported = None
    trt_engine = None
    try:
        exported = torch.export.export(
            mot_step,
            args=(latents_eager, timestep),
            strict=False,
        )
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=make_input_spec((latents_eager, timestep)),
            **TRT_SETTINGS,
        )
        with torch.no_grad():
            vel_trt = trt_engine(latents_eager, timestep)
    finally:
        free_cuda_memory(trt_engine, exported, mot_step, transformer)

    return 0


if __name__ == "__main__":
    SystemExit(main())
