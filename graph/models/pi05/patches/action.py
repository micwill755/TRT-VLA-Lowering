"""PI05 action-expert wrap for ``EdgeExporter.export()``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from . import root


def wrap_action(policy: nn.Module, _inputs: Mapping[str, Any] | None = None) -> nn.Module:
    from trt.modules.export.diffusion import (
        PI05PrefixKVStepEncoderExportModule,
        StaticActionVelocityStepExportModule,
    )

    found = root(policy)
    if found is None:
        raise RuntimeError("PI05 action wrap matched a policy without paligemma_with_expert")
    return StaticActionVelocityStepExportModule(
        step_encoder=PI05PrefixKVStepEncoderExportModule(found),
        action_expert=found.paligemma_with_expert.gemma_expert.model,
        velocity_decoder=found.action_out_proj,
        output_tokens=int(found.config.chunk_size),
        cast_hidden_fp32=False,
    ).eval()
