"""GR00T action-context and action-expert wraps for ``EdgeExporter.export()``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from . import root


def wrap_action_context(policy: nn.Module, _inputs: Mapping[str, Any] | None = None) -> nn.Module:
    from trt.modules.export.language import ContextProjectionExportModule

    found = root(policy)
    if found is None:
        raise RuntimeError("GR00T action_context wrap matched a policy without backbone")
    return ContextProjectionExportModule(
        found.backbone.eagle_linear,
        found.action_head.vlln,
        found.action_head.vl_self_attention,
    ).eval()


def wrap_action(policy: nn.Module, inputs: Mapping[str, Any] | None = None) -> nn.Module:
    from trt.modules.export.diffusion import (
        GrootDiTStepEncoderExportModule,
        StaticActionVelocityStepExportModule,
        TRTDynamicCategorySpecificMLPExportModule,
    )

    found = root(policy)
    if found is None:
        raise RuntimeError("GR00T action wrap matched a policy without action_head")
    action_head = found.action_head
    embodiment_id = None if inputs is None else inputs.get("embodiment_id")
    return StaticActionVelocityStepExportModule(
        step_encoder=GrootDiTStepEncoderExportModule(action_head, embodiment_id),
        action_expert=action_head.model,
        velocity_decoder=TRTDynamicCategorySpecificMLPExportModule(
            action_head.action_decoder
        ),
        output_tokens=int(action_head.config.action_horizon),
        cast_hidden_fp32=False,
    ).eval()
