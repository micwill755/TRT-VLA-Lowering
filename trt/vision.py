"""Vision tower export helpers for Edge-LLM VitRunner.

Layout conventions
------------------
* **Policy / HuggingFace**: NCHW ``[batch, C, H, W]`` (LeRobot processor output).
* **TRT engine / VitRunner**: HWC ``[batch, H, W, C]`` binding ``pixel_values``.
* **Engine output**: flattened ``[batch * num_tokens, hidden]`` for C++
  ``embeddingLookupWithImageInsertion``;.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from trt.compile import save_trt_engine_module
from trt.io_spec import ComponentIOSpec, VLA_VISION_IO
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.utils import free_cuda_memory

logger = logging.getLogger(__name__)

# Canonical VitRunner engine binding names (shared by PI0.5 and GR00T).
VIT_ENGINE_INPUT_NAME = VLA_VISION_IO.input_names[0]
VIT_ENGINE_OUTPUT_NAME = VLA_VISION_IO.output_names[0]

DEFAULT_VISION_TRT_SETTINGS: dict[str, Any] = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
}


@dataclass
class VisionEngineSpec:
    """Model-specific vision export configuration for ``save_visual_engine_for_edge_llm``."""

    visual_vision_model: nn.Module
    patch_vision_model: nn.Module
    projector: nn.Module

    input_dtype: torch.dtype
    patch_batch_size: int
    patch_seq_len: int
    vocab_size: int
    image_token_id: int
    config_seq_len: int = 0
    output_hidden_size: int = 0
    # VitRunner flat output [B*S, H] and per-image [B, S, H]; set by save_visual_engine_for_edge_llm.
    image_embed_flat_shape: tuple[int, int] = ()
    image_embed_shape: tuple[int, int, int] = ()

    select_layer: int = -1
    pixel_shuffle: bool = False
    downsample_ratio: float = 0.5
    force_float32_input: bool = False
    cast_output_to_input_dtype: bool = False
    vision_kwargs: dict[str, Any] = field(default_factory=dict)

    patch_name: str = "SigLIP"
    allow_attention_mask: bool = False

    io: ComponentIOSpec = VLA_VISION_IO
    trt_settings: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_VISION_TRT_SETTINGS))
    model_type: str = "vit"

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