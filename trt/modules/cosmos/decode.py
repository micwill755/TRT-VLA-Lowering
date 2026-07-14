from __future__ import annotations

import torch
import torch.nn as nn

from trt.modules.cosmos.packing import decode_cosmos_video


class CosmosVaeDecodeExportModule(nn.Module):
    """normalized latents [B,C,T,H,W] -> pixels [B,3,T,H,W]."""

    def __init__(self, vae: nn.Module, sample_latents: torch.Tensor):
        super().__init__()
        self.vae = vae
        dtype = vae.dtype
        mean = torch.tensor(vae.config.latents_mean, dtype=dtype)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=dtype)
        self.register_buffer("latents_mean", mean.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_inv_std", inv_std.view(1, -1, 1, 1, 1), persistent=False)

        with torch.no_grad():
            pixels = self.forward(sample_latents)
            self.pixel_shape = tuple(pixels.shape)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return decode_cosmos_video(self.vae, latents)
