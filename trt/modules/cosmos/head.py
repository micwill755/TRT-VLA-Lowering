from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Cosmos3VisionDenoiseHeadExportModule(nn.Module):
    """Vision denoise head: last_hidden_state -> pred vision latents.

    Engine 4 (Edge). proj_out on noisy slots + unpatchify.
    IN:  last_hidden_state [sequence_length, hidden]
    OUT: pred_vision_latents [B, C, T, H, W]
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        packed_static: dict[str, Any],
        sample_latents: torch.Tensor,
        sample_last_hidden: torch.Tensor,
    ):
        super().__init__()
        self.transformer = transformer
        self.proj_out = transformer.proj_out
        self.vision_token_shapes = packed_static["vision_token_shapes"]
        self.vision_noisy_frame_indexes = packed_static["vision_noisy_frame_indexes"]
        self.register_buffer(
            "vision_mse_loss_indexes",
            packed_static["vision_mse_loss_indexes"],
            persistent=False,
        )

        with torch.no_grad():
            _, original_latent_shapes = transformer._patchify_and_pack_latents([sample_latents])
        self.original_latent_shapes = original_latent_shapes

        with torch.no_grad():
            out = self.forward(sample_last_hidden)
            self.output_shape = tuple(out.shape)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        preds_packed = self.proj_out(last_hidden_state[self.vision_mse_loss_indexes])
        preds = self.transformer._unpatchify_and_unpack_latents(
            preds_packed,
            token_shapes_vision=self.vision_token_shapes,
            noisy_frame_indexes_vision=self.vision_noisy_frame_indexes,
            original_latent_shapes=self.original_latent_shapes,
        )
        return preds[0]


class Cosmos3SoundDenoiseHeadExportModule(nn.Module):
    """Sound denoise head: last_hidden_state -> pred sound latents [C, T]."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        packed_static: dict[str, Any],
        sample_last_hidden: torch.Tensor,
    ):
        super().__init__()
        self.transformer = transformer
        self.audio_proj_out = transformer.audio_proj_out
        self.sound_token_shapes = packed_static["sound_token_shapes"]
        self.sound_noisy_frame_indexes = packed_static["sound_noisy_frame_indexes"]
        self.register_buffer(
            "sound_mse_loss_indexes",
            packed_static["sound_mse_loss_indexes"],
            persistent=False,
        )

        with torch.no_grad():
            out = self.forward(sample_last_hidden)
            self.output_shape = tuple(out.shape)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        preds_packed = self.audio_proj_out(last_hidden_state[self.sound_mse_loss_indexes])
        preds = self.transformer._unpack_sound_latents(
            preds_packed,
            self.sound_token_shapes,
            self.sound_noisy_frame_indexes,
        )
        return preds[0]
