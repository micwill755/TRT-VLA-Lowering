from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _embed_timestep(transformer: nn.Module, timesteps: torch.Tensor) -> torch.Tensor:
    """Run ``time_proj`` + ``time_embedder`` with dtype alignment for fp16 export."""
    embed_dtype = next(transformer.time_embedder.parameters()).dtype
    projected = transformer.time_proj(timesteps).to(dtype=embed_dtype)
    return transformer.time_embedder(projected)


class Cosmos3VisionGenEmbedExportModule(nn.Module):
    """Vision branch of gen_seq embed: patchify + proj_in + timestep on noisy slots.

    Edge engine 2 (vision-only). Omni extends with sound/action branches later.
    IN:  vision_latents [B,C,T,H,W], scalar timestep
    OUT: gen_seq_vision [num_vision_tokens, hidden]
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        packed_static: dict[str, Any],
        sample_latents: torch.Tensor,
        sample_timestep: float,
    ):
        super().__init__()
        self.transformer = transformer
        self.vision_token_shapes = packed_static["vision_token_shapes"]
        self.vision_noisy_frame_indexes = packed_static["vision_noisy_frame_indexes"]
        self.num_noisy_tokens = int(packed_static["num_noisy_tokens"])
        self.timestep_scale = float(transformer.config.timestep_scale)

        with torch.no_grad():
            t = torch.tensor(sample_timestep, device=sample_latents.device, dtype=sample_latents.dtype)
            out = self.forward(sample_latents, t)
            self.output_shape = tuple(out.shape)

    def forward(self, vision_latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        transformer = self.transformer
        packed_tokens_vision, _ = transformer._patchify_and_pack_latents([vision_latents])
        packed_tokens_vision = transformer.proj_in(packed_tokens_vision)

        timestep = timestep.to(device=vision_latents.device, dtype=vision_latents.dtype).reshape(())
        vision_timesteps = (timestep.expand(self.num_noisy_tokens) * self.timestep_scale).to(
            dtype=vision_latents.dtype
        )
        packed_timestep_embeds = _embed_timestep(transformer, vision_timesteps)
        packed_timestep_embeds = packed_timestep_embeds.to(dtype=packed_tokens_vision.dtype)
        return transformer._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds,
            noisy_frame_indexes=self.vision_noisy_frame_indexes,
            token_shapes=self.vision_token_shapes,
        )


class Cosmos3OmniGenEmbedExportModule(nn.Module):
    """Omni gen_seq embed: vision + sound branches fused for the MoT backbone.

    IN:  vision_latents [B,C,T,H,W], sound_latents [C,T], scalar timestep
    OUT: gen_seq [num_vision_tokens + sound_len, hidden]
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        packed_static: dict[str, Any],
        sample_vision_latents: torch.Tensor,
        sample_sound_latents: torch.Tensor,
        sample_timestep: float,
    ):
        super().__init__()
        self.transformer = transformer
        self.vision_token_shapes = packed_static["vision_token_shapes"]
        self.vision_noisy_frame_indexes = packed_static["vision_noisy_frame_indexes"]
        self.num_noisy_vision_tokens = int(packed_static["num_noisy_tokens"])
        self.sound_token_shapes = packed_static["sound_token_shapes"]
        self.sound_noisy_frame_indexes = packed_static["sound_noisy_frame_indexes"]
        self.num_noisy_sound_tokens = int(packed_static["num_noisy_sound_tokens"])
        self.timestep_scale = float(transformer.config.timestep_scale)

        with torch.no_grad():
            t = torch.tensor(sample_timestep, device=sample_vision_latents.device, dtype=sample_vision_latents.dtype)
            out = self.forward(sample_vision_latents, sample_sound_latents, t)
            self.output_shape = tuple(out.shape)

    def forward(
        self,
        vision_latents: torch.Tensor,
        sound_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        transformer = self.transformer
        packed_tokens_vision, _ = transformer._patchify_and_pack_latents([vision_latents])
        packed_tokens_vision = transformer.proj_in(packed_tokens_vision)

        timestep = timestep.to(device=vision_latents.device, dtype=vision_latents.dtype).reshape(())
        vision_timesteps = (timestep.expand(self.num_noisy_vision_tokens) * self.timestep_scale).to(
            dtype=vision_latents.dtype
        )
        packed_timestep_embeds_vision = _embed_timestep(transformer, vision_timesteps)
        packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(dtype=packed_tokens_vision.dtype)
        packed_tokens_vision = transformer._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=self.vision_noisy_frame_indexes,
            token_shapes=self.vision_token_shapes,
        )

        packed_tokens_sound = transformer._pack_sound_latents([sound_latents], self.sound_token_shapes).to(
            packed_tokens_vision.dtype
        )
        packed_tokens_sound = transformer.audio_proj_in(packed_tokens_sound) + transformer.audio_modality_embed
        sound_timesteps = (timestep.expand(self.num_noisy_sound_tokens) * self.timestep_scale).to(
            dtype=vision_latents.dtype
        )
        packed_timestep_embeds_sound = _embed_timestep(transformer, sound_timesteps)
        packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(dtype=packed_tokens_sound.dtype)
        packed_tokens_sound = transformer._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_sound,
            packed_timestep_embeds=packed_timestep_embeds_sound,
            noisy_frame_indexes=self.sound_noisy_frame_indexes,
            token_shapes=self.sound_token_shapes,
        )
        return torch.cat([packed_tokens_vision, packed_tokens_sound], dim=0)
