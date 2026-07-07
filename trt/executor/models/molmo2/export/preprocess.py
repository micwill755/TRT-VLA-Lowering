from __future__ import annotations

from trt.context import EdgeContext


def preprocess(ctx: EdgeContext) -> None:
    """Normalize model inputs before the stage loop (MolmoAct2)."""
    # TODO: port MolmoAct2ExportHooks.preprocess logic here.
    _ = ctx
