from __future__ import annotations

import torch
import torch.nn as nn


class CosmosAvaeEncodeExportModule(nn.Module):
    """waveform [B,C,N] -> sound latents [B,C,T]."""

    def __init__(self, sound_tokenizer: nn.Module, sample_waveform: torch.Tensor):
        super().__init__()
        self.sound_tokenizer = sound_tokenizer

        with torch.no_grad():
            latents = self.forward(sample_waveform)
            self.latent_shape = tuple(latents.shape)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        in_dtype = waveform.dtype
        encoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        encoded = self.sound_tokenizer.encode(waveform.to(dtype=encoder_dtype), return_dict=True)
        return encoded.latent_dist.mode().to(in_dtype)


class CosmosAvaeDecodeExportModule(nn.Module):
    """sound latents [B,C,T] -> waveform [B,audio_ch,N]."""

    def __init__(self, sound_tokenizer: nn.Module, sample_latents: torch.Tensor):
        super().__init__()
        self.sound_tokenizer = sound_tokenizer

        with torch.no_grad():
            waveform = self.forward(sample_latents)
            self.waveform_shape = tuple(waveform.shape)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        return self.sound_tokenizer.decode(latents.to(dtype=decoder_dtype))
