"""Minimal functions reused from the Cosmos3-Edge WFM export path."""

from __future__ import annotations

import logging

import torch
import torch_tensorrt
from accelerate.utils import set_module_tensor_to_device
from diffusers import Cosmos3OmniPipeline

torch_tensorrt.logging.set_level(logging.WARNING)

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}


def fix_edge_diffusers_weights(
    transformer,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Materialize Edge checkpoint tensors that Diffusers leaves on meta."""
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
