"""Vision tower export helpers for Edge-LLM VitRunner.

Layout conventions
------------------
* **Policy / HuggingFace**: NCHW ``[batch, C, H, W]`` (LeRobot processor output).
* **TRT engine / VitRunner**: HWC ``[batch, H, W, C]`` binding ``pixel_values``.
* **Engine output**: flattened ``[batch * num_tokens, hidden]`` for C++
  ``embeddingLookupWithImageInsertion``; PI0.5 prefix packing reshapes back to
  ``[batch, num_tokens, hidden]`` inside ``prepare_compact_prefix``.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from trt.io_spec import VLA_VISION_IO

logger = logging.getLogger(__name__)

# Canonical VitRunner engine binding names (shared by PI0.5 and GR00T).
VIT_ENGINE_INPUT_NAME = VLA_VISION_IO.input_names[0]
VIT_ENGINE_OUTPUT_NAME = VLA_VISION_IO.output_names[0]


def nchw_to_hwc(pixel_values: torch.Tensor) -> torch.Tensor:
    """Convert LeRobot/HF NCHW pixels to VitRunner HWC layout.

    Args:
        pixel_values: ``[batch, C, H, W]`` float tensor from the policy processor.

    Returns:
        ``[batch, H, W, C]`` contiguous tensor for ``VisualFixedInput`` / TRT export.
    """
    if pixel_values.ndim != 4:
        raise ValueError(f"Expected 4D pixel_values, got shape {tuple(pixel_values.shape)}")
    return pixel_values.permute(0, 2, 3, 1).contiguous()


def hwc_to_nchw(images: torch.Tensor) -> torch.Tensor:
    """Convert VitRunner HWC pixels to HuggingFace SigLIP NCHW layout.

    Args:
        images: ``[batch, H, W, C]`` tensor at the TRT/C++ boundary.

    Returns:
        ``[batch, C, H, W]`` contiguous tensor for ``vision_model(pixel_values=...)``.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D images, got shape {tuple(images.shape)}")
    return images.permute(0, 3, 1, 2).contiguous()


def is_nchw_pixel_values(pixel_values: torch.Tensor) -> bool:
    """Return True when channels are in dim 1 (processor-style NCHW).

    Used inside ``VisualFixedInput._run_vision`` to avoid double permuting tensors
    that are already in the layout expected by HuggingFace vision towers.
    """
    return (
        pixel_values.ndim == 4
        and pixel_values.shape[1] in (1, 3, 4)
        and pixel_values.shape[-1] not in (1, 3, 4)
    )


def run_trt_vision_nchw(trt_vision: nn.Module, pixel_values_nchw: torch.Tensor) -> torch.Tensor:
    """Run an in-memory TRT vision module from policy-style NCHW input.

    The compiled module expects HWC ``pixel_values``; this helper performs the
    layout conversion and returns VitRunner-style flattened embeds
    ``[batch * num_tokens, hidden]``.
    """
    return trt_vision(nchw_to_hwc(pixel_values_nchw.contiguous()))


def vit_visual_edge_config(
    *,
    vocab_size: int,
    image_token_id: int,
    seq_len: int,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> dict[str, Any]:
    """Build VitRunner metadata merged into ``visual/config.json`` on export.

    These fields are read by C++ ``VitRunner`` at runtime (not part of the TRT
    graph). They tell ``llm_inference`` how to expand ``<image>`` placeholders in
    the prompt and wire ``visual_embeds`` rows into the LM embedding table.

    Args:
        vocab_size: LM vocab size; synthetic image slot IDs start here.
        image_token_id: Token ID in the chat template replaced by ``seq_len`` vision rows.
        seq_len: Number of vision tokens produced per image (e.g. 256 for SigLIP PI0.5).
        image_mean: Optional per-channel mean for C++ ``normalizeImage`` (RGB order).
        image_std: Optional per-channel std for C++ ``normalizeImage`` (RGB order).

    Returns:
        Dict passed as ``extra_config`` to ``save_trt_engine_module``.
    """
    builder_config: dict[str, Any] = {"seq_len": int(seq_len)}
    if image_mean is not None:
        builder_config["image_mean"] = list(image_mean)
    if image_std is not None:
        builder_config["image_std"] = list(image_std)
    return {
        "vocab_size": int(vocab_size),
        "image_token_id": int(image_token_id),
        "builder_config": builder_config,
    }


class VisualFixedInput(nn.Module):
    """Fixed-shape vision tower + projector wrapper for TRT export.

    Expects **preprocessed** HWC input ``[batch, H, W, C]`` (matches C++
    ``VitRunner::normalizeImage`` output), not raw uint8 images. The TRT engine
    output is flattened to ``[batch * output_seq_len, lm_hidden]`` so C++ can
    index vision rows sequentially via ``embeddingLookupWithImageInsertion``.

    Typical flow (GR00T may stack multiple cameras on the batch dim)::

        pixel_values  [batch, H, W, C]   e.g. [2, 224, 224, 3]
            -> vision_model (SigLIP, internal NCHW)
        vit_embeds      [batch, seq_len, vit_hidden]
            -> optional pixel_shuffle (Eagle / Qwen-style downsampling)
            -> projector
        features        [batch, output_seq_len, lm_hidden]
            -> reshape
        return          [batch * output_seq_len, lm_hidden]
    """

    def __init__(
        self,
        *,
        vision_model: nn.Module,
        projector: nn.Module,
        sample_pixel_values: torch.Tensor,
        select_layer: int = -1,
        pixel_shuffle: bool = False,
        downsample_ratio: float = 0.5,
        force_float32_input: bool = False,
        cast_output_to_input_dtype: bool = False,
        vision_kwargs: dict[str, Any] | None = None,
    ):
        """Probe output shapes once so TRT export uses fixed static dimensions.

        Args:
            vision_model: HF vision tower (e.g. SigLIP ``vision_model``).
            projector: Maps vision hidden states to LM hidden size.
            sample_pixel_values: Representative HWC input ``[batch, H, W, C]``.
            select_layer: Hidden-state index; ``-1`` uses ``last_hidden_state``.
            pixel_shuffle: Apply spatial downsampling before projection (Eagle models).
            downsample_ratio: Shuffle ratio when ``pixel_shuffle`` is enabled.
            force_float32_input: Cast input to FP32 inside forward (SigLIP export path).
            cast_output_to_input_dtype: Cast projector output back to input dtype.
            vision_kwargs: Extra kwargs forwarded to ``vision_model`` (e.g. ``image_grid_thw``).
        """
        super().__init__()

        self.vision_model = vision_model
        self.projector = projector
        self.select_layer = int(select_layer)
        self.pixel_shuffle = bool(pixel_shuffle)
        self.downsample_ratio = float(downsample_ratio)
        self.force_float32_input = bool(force_float32_input)
        self.cast_output_to_input_dtype = bool(cast_output_to_input_dtype)
        self.vision_kwargs = dict(vision_kwargs or {})

        with torch.no_grad():
            sample = sample_pixel_values  # [batch, H, W, C]
            if self.force_float32_input and sample.dtype != torch.float32:
                # TODO, look into method for removing forced fp32 conversion.
                sample = sample.to(torch.float32)

            vit_embeds = self._select_vision_features(self._run_vision(sample))
            # vit_embeds: [batch, seq_len, vit_hidden]

            self.seq_len = int(vit_embeds.shape[1])
            self.hidden_size = int(vit_embeds.shape[2])

            if self.pixel_shuffle:
                self._init_pixel_shuffle_shape()
                vit_embeds = self._apply_pixel_shuffle(vit_embeds)
                # vit_embeds: [batch, output_seq_len, shuffle_hidden]

            projected = self.projector(vit_embeds)
            # projected: [batch, output_seq_len, lm_hidden]
            self.batch_size = int(projected.shape[0])
            self.output_seq_len = int(projected.shape[1])
            self.output_hidden_size = int(projected.shape[2])
            self.output_num_tokens = self.batch_size * self.output_seq_len

    def _run_vision(self, images: torch.Tensor):
        """Call the HF vision tower, converting HWC -> NCHW when needed."""
        # TRT / VitRunner boundary is HWC; HuggingFace SigLIP expects NCHW.
        pixel_values = hwc_to_nchw(images) if not is_nchw_pixel_values(images) else images
        kwargs = dict(self.vision_kwargs)
        kwargs["pixel_values"] = pixel_values
        kwargs["output_hidden_states"] = self.select_layer != -1
        kwargs.setdefault("return_dict", True)
        return self.vision_model(**kwargs)

    def _select_vision_features(self, out):
        """Pick last hidden state or an intermediate layer from the vision output."""
        if self.select_layer == -1:
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return out.hidden_states[self.select_layer]

    def _init_pixel_shuffle_shape(self):
        """Derive grid dimensions for pixel-shuffle downsampling from ``seq_len``."""
        side = int(self.seq_len ** 0.5)
        if side * side != self.seq_len:
            raise ValueError(f"Expected square vision sequence, got seq_len={self.seq_len}")

        self.grid_w = side
        self.grid_h = side
        self.out_w = int(self.grid_w * self.downsample_ratio)
        self.out_h = int(self.grid_h * self.downsample_ratio)
        self.hidden_after_first_view = int(self.hidden_size / self.downsample_ratio)
        self.shuffle_hidden = int(
            self.hidden_size / (self.downsample_ratio * self.downsample_ratio)
        )

    def _apply_pixel_shuffle(self, x):
        """Spatially downsample vision tokens before projection (Eagle-style)."""
        # x: [batch, seq_len, vit_hidden]  where seq_len = grid_h * grid_w
        n = x.shape[0]

        x = x.reshape(n, self.grid_w, self.out_h, self.hidden_after_first_view)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.reshape(n, self.out_h, self.out_w, self.shuffle_hidden)
        x = x.permute(0, 2, 1, 3).contiguous()

        # [batch, output_seq_len, shuffle_hidden]
        return x.reshape(n, -1, self.shuffle_hidden)

    def forward(self, pixel_values):
        """Run vision + projector; return VitRunner-style flattened embeds.

        Args:
            pixel_values: Preprocessed HWC ``[batch, H, W, C]`` (TRT binding name).

        Returns:
            ``[batch * output_seq_len, lm_hidden]`` for Edge-LLM engine export.
        """
        images = pixel_values
        out_dtype = images.dtype

        if self.force_float32_input and images.dtype != torch.float32:
            # TODO, look into method for removing forced fp32 conversion.
            images = images.to(torch.float32)

        vit_embeds = self._select_vision_features(self._run_vision(images))
        # vit_embeds: [batch, seq_len, vit_hidden]

        if self.pixel_shuffle:
            vit_embeds = self._apply_pixel_shuffle(vit_embeds)
            # vit_embeds: [batch, output_seq_len, shuffle_hidden]

        features = self.projector(vit_embeds)
        # features: [batch, output_seq_len, lm_hidden]

        if self.cast_output_to_input_dtype and features.dtype != out_dtype:
            features = features.to(out_dtype)

        # Flatten batch + tokens for Edge-LLM: [batch * output_seq_len, lm_hidden]
        return features.reshape(-1, features.shape[-1])
