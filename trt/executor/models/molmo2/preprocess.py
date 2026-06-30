from __future__ import annotations

from trt.runner.base import StageContext


def preprocess(ctx: StageContext) -> None:
    """Normalize model inputs before the stage loop (MolmoAct2)."""
    # TODO: port MolmoAct2ExportHooks.preprocess logic here.
    _ = ctx
