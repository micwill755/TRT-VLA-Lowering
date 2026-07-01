from __future__ import annotations

from trt.runner.base import StageContext


def plan_export(ctx: StageContext, stage_inputs: dict):
    """Action/diffusion stage export (stub — use legacy export until implemented)."""
    del ctx, stage_inputs
    raise NotImplementedError("GR00T staged action export is not implemented; use use_legacy=True")
