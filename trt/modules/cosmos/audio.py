from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm


def build_stft_dft_conv_weight(n_fft: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Fixed conv1d kernel that reproduces ``torch.stft`` (Hann window, onesided, center=False).

    ``torch.stft`` with a fixed window/``n_fft`` is a linear op: each frame is windowed then hit with
    the DFT basis. That is exactly a ``conv1d`` (cross-correlation) with precomputed cos/sin kernels,
    so it lowers cleanly in TensorRT (no FFT / complex tensors). Output channel layout matches the
    encoder's ``cat([real, imag], dim=1)``: bins ``[0, n_fft//2]`` real, then the same bins imaginary.
    """
    num_bins = n_fft // 2 + 1
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
    n = torch.arange(n_fft, dtype=torch.float32)
    k = torch.arange(num_bins, dtype=torch.float32).unsqueeze(1)
    angle = 2.0 * math.pi * k * n / n_fft  # [num_bins, n_fft]
    real_kernel = window.unsqueeze(0) * torch.cos(angle)
    imag_kernel = -window.unsqueeze(0) * torch.sin(angle)
    weight = torch.cat([real_kernel, imag_kernel], dim=0).unsqueeze(1)  # [2*num_bins, 1, n_fft]
    return weight.to(dtype)


def fold_avae_decoder_weight_norm(decoder: nn.Module) -> nn.Module:
    """Return a deep copy of the Oobleck decoder with ``weight_norm`` folded into static conv weights.

    TensorRT fails to build the original graph because ``weight_norm`` recomputes normalized weights at
    runtime and bf16 ``ConvTranspose1d`` tactics are missing on some stacks. Folding produces a plain
    conv/deconv graph with immutable weights.
    """
    folded = copy.deepcopy(decoder)
    for mod in folded.modules():
        if hasattr(mod, "weight_g"):
            remove_weight_norm(mod)
    return folded.eval()


def build_trt_avae_decoder(decoder: nn.Module, *, dtype: torch.dtype = torch.float32) -> nn.Module:
    """Fold weight_norm and cast to ``dtype`` (fp32 by default) for TensorRT engine build."""
    return fold_avae_decoder_weight_norm(decoder).to(dtype=dtype)


def prepare_avae_waveform_for_encode(waveform: torch.Tensor, sound_tokenizer: nn.Module) -> torch.Tensor:
    """Normalize and pad waveform the same way as ``sound_tokenizer.encode()``."""
    hidden_states = waveform
    if sound_tokenizer.config.normalize_volume:
        hidden_states = hidden_states / (hidden_states.abs().max() + 1e-5) * 0.95

    hop_size = int(sound_tokenizer._hop_size)
    sample_length = hidden_states.shape[-1]
    padding = (hop_size - (sample_length % hop_size)) % hop_size
    if padding > 0:
        hidden_states = F.pad(hidden_states, (0, padding), mode="constant", value=0)

    encoder_dtype = next(sound_tokenizer.encoder.parameters()).dtype
    return hidden_states.to(dtype=encoder_dtype)


def compute_avae_spectrogram(waveform: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    """STFT spectrogram front-end from ``Cosmos3AudioSpectrogramConvNeXtEncoder``.

    This path uses ``torch.stft`` / FFT and is not TRT-compilable; keep it eager and compile only
    the ConvNeXt stack that follows.
    """
    batch_size, num_channels, _num_samples = waveform.shape
    audio = waveform
    if num_channels > 1:
        audio = audio.reshape(batch_size * num_channels, 1, _num_samples)

    spectrogram = encoder._spectrogram(audio.squeeze(1))
    real, imaginary = torch.view_as_real(spectrogram).chunk(2, dim=-1)
    spectrogram = torch.cat([real, imaginary], dim=1).squeeze(-1)
    spectrogram = spectrogram.to(audio.dtype)
    if num_channels > 1:
        spectrogram = spectrogram.reshape(batch_size, num_channels * spectrogram.shape[1], spectrogram.shape[2])
    return spectrogram


class CosmosAvaeEncoderCoreExportModule(nn.Module):
    """spectrogram [B,C,T] -> sound latents [B,C',T'] (ConvNeXt stack only, no STFT)."""

    def __init__(self, sound_tokenizer: nn.Module, sample_spectrogram: torch.Tensor):
        super().__init__()
        self.encoder = sound_tokenizer.encoder

        with torch.no_grad():
            latents = self.forward(sample_spectrogram)
            self.latent_shape = tuple(latents.shape)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        in_dtype = spectrogram.dtype
        moments = self.encoder.layers(spectrogram)
        mean, _scale = moments.chunk(2, dim=1)
        return mean.to(in_dtype)


class CosmosAvaeEncodeConvStftExportModule(nn.Module):
    """waveform [B,C,N] -> sound latents [B,C',T'] with a conv-based STFT (TRT-compilable, no FFT).

    Reproduces ``Cosmos3AVAEAudioTokenizer.encode`` + ``Cosmos3AudioSpectrogramConvNeXtEncoder`` but
    replaces the ``torch.stft`` front-end with a fixed ``conv1d`` DFT bank so the whole
    waveform->latent path compiles into a single TensorRT engine.
    """

    def __init__(self, sound_tokenizer: nn.Module, sample_waveform: torch.Tensor):
        super().__init__()
        encoder = sound_tokenizer.encoder
        if encoder is None:
            raise ValueError("sound_tokenizer has no encoder; cannot build an AVAE encode engine.")
        self.encoder = encoder
        self.layers = encoder.layers
        self.input_channels = int(encoder.input_channels)
        self.n_fft = int(encoder.n_fft)
        self.hop_length = int(encoder.hop_length)
        self.normalize_volume = bool(sound_tokenizer.config.normalize_volume)
        self.hop_size = int(sound_tokenizer._hop_size)

        self.register_buffer(
            "dft_weight", build_stft_dft_conv_weight(self.n_fft), persistent=False
        )

        with torch.no_grad():
            latents = self.forward(sample_waveform)
            self.latent_shape = tuple(latents.shape)

    def _conv_spectrogram(self, audio_2d: torch.Tensor) -> torch.Tensor:
        # Mirror Cosmos3AudioSpectrogramConvNeXtEncoder._spectrogram padding, in fp32.
        pad_left = (self.n_fft - self.hop_length) // 2
        pad_right = (self.n_fft - self.hop_length) - pad_left
        x = F.pad(audio_2d, (pad_left, pad_right)).unsqueeze(1).float()
        return F.conv1d(x, self.dft_weight.to(x.device), stride=self.hop_length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        in_dtype = waveform.dtype
        hidden = waveform
        if self.normalize_volume:
            hidden = hidden / (hidden.abs().max() + 1e-5) * 0.95

        sample_length = hidden.shape[-1]
        padding = (self.hop_size - (sample_length % self.hop_size)) % self.hop_size
        if padding > 0:
            hidden = F.pad(hidden, (0, padding), mode="constant", value=0)

        encoder_dtype = next(self.layers.parameters()).dtype
        batch_size, num_channels, num_samples = hidden.shape
        audio = hidden
        if num_channels > 1:
            audio = audio.reshape(batch_size * num_channels, 1, num_samples)

        spectrogram = self._conv_spectrogram(audio.squeeze(1)).to(encoder_dtype)
        if num_channels > 1:
            spectrogram = spectrogram.reshape(
                batch_size, num_channels * spectrogram.shape[1], spectrogram.shape[2]
            )

        moments = self.layers(spectrogram)
        mean, _scale = moments.chunk(2, dim=1)
        return mean.to(in_dtype)


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


class CosmosAvaeDecodeTrtExportModule(nn.Module):
    """sound latents [B,C,T] -> waveform [B,audio_ch,N] (TRT-compilable Oobleck decoder).

    Folds ``weight_norm`` into static conv/deconv weights and runs the decoder in fp32 so TensorRT
    can select deconv tactics. Output is cast back to the input latent dtype.
    """

    def __init__(self, sound_tokenizer: nn.Module, sample_latents: torch.Tensor):
        super().__init__()
        self.decoder = build_trt_avae_decoder(sound_tokenizer.decoder, dtype=torch.float32)

        with torch.no_grad():
            waveform = self.forward(sample_latents)
            self.waveform_shape = tuple(waveform.shape)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        in_dtype = latents.dtype
        audio = self.decoder(latents.float()).clamp(-1.0, 1.0)
        return audio.to(in_dtype)
