from __future__ import annotations

import torch
import torch.nn as nn

from trt.modules.cosmos.packing import encode_cosmos_video
from trt.modules.export.vision import ExportModule


class CosmosVaeEncodeExportModule(nn.Module):
    """pixels [B,3,T,H,W] -> normalized Cosmos latents."""

    def __init__(self, vae: nn.Module, sample_pixels: torch.Tensor):
        super().__init__()
        self.vae = vae

        with torch.no_grad():
            latents = self.forward(sample_pixels)
            self.latent_shape = tuple(latents.shape)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return encode_cosmos_video(self.vae, pixels)

class Cosmos3MoTDenoiseStepExportModule(nn.Module):
    """One scheduler step through Cosmos3OmniTransformer.

    Static packed-sequence metadata (indexes, shapes, position_ids) is fixed at
    export time. Per-step inputs are noisy latents + timestep.
  """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        packed_static: dict[str, torch.Tensor | int | list],
        sample_latents: torch.Tensor,
        sample_timestep: float,
        modality: str = "vision",   # "vision" | "action" | "vision+action"
        domain_id: torch.Tensor | None = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.packed_static = packed_static
        self.modality = modality
        self.domain_id = domain_id

        with torch.no_grad():
            t = torch.tensor(sample_timestep, device=sample_latents.device, dtype=sample_latents.dtype)
            out = self._step(sample_latents, t)
            self.output_shape = tuple(out.shape)

    def _step(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        ps = self.packed_static
        num_noisy = int(ps["num_noisy_tokens"])
        timestep = timestep.to(device=latents.device, dtype=latents.dtype).reshape(())
        ts = timestep.expand(num_noisy)

        kwargs = dict(
            input_ids=ps["input_ids"],
            text_indexes=ps["text_indexes"],
            position_ids=ps["position_ids"],
            und_len=int(ps["und_len"]),
            sequence_length=int(ps["sequence_length"]),
            vision_tokens=[latents],
            vision_token_shapes=ps["vision_token_shapes"],
            vision_sequence_indexes=ps["vision_sequence_indexes"],
            vision_mse_loss_indexes=ps["vision_mse_loss_indexes"],
            vision_timesteps=ts,
            vision_noisy_frame_indexes=ps["vision_noisy_frame_indexes"],
        )

        if "action" in self.modality:
            action_latents = ps["sample_action_latents"]
            kwargs.update(
                action_tokens=[action_latents],
                action_token_shapes=ps["action_token_shapes"],
                action_sequence_indexes=ps["action_sequence_indexes"],
                action_mse_loss_indexes=ps["action_mse_loss_indexes"],
                action_timesteps=timestep.expand(int(ps["action_noisy_len"])),
                action_noisy_frame_indexes=ps["action_noisy_frame_indexes"],
                action_domain_ids=[self.domain_id],
            )

        preds_vision, _, preds_action = self.transformer(**kwargs)

        if self.modality == "action":
            return preds_action[0]
        return preds_vision[0]

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self._step(latents, timestep)