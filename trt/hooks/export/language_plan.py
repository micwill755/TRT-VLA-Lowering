from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from trt.hooks.export.base import ExportPlanBase
from trt.io_spec import ComponentIOSpec, VLA_LANGUAGE_IO


@dataclass
class LanguageExportPlan(ExportPlanBase):
    module: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    decoder: nn.Module
    language_model: nn.Module
    language_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    batch_size: int = 1
    max_seq_len: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    image_token_id: int = 0
    seq_len_per_image: int = 0
    select_layer: int = -1
    enable_bidirectional_prefill: int = 0
    static_prefill_seq_len: bool = False
    export_dtype: torch.dtype = torch.float16
    io: ComponentIOSpec = VLA_LANGUAGE_IO
    context_hidden_size: int | None = None
