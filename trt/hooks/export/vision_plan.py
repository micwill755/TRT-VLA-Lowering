from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trt.hooks.export.base import ExportPlanBase


@dataclass
class VisionExportPlan(ExportPlanBase):
    module: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    patch_target: nn.Module
    patch_batch_size: int
    patch_seq_len: int
    vocab_size: int
    image_token_id: int
    config_seq_len: int
    patch_name: str = "vision"
    allow_attention_mask: bool = False
