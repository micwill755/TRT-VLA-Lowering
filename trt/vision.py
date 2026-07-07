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
from typing import Any

import torch
import torch.nn as nn

from trt.io_spec import VLA_VISION_IO

logger = logging.getLogger(__name__)

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