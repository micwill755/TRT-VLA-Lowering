from __future__ import annotations

from trt.runner.base import StageContext


def postprocess(ctx: StageContext) -> None:
    """End-to-end parity checks after all stages complete (GR00T)."""
    # TODO: port GrootExportHooks.after_export parity logic here.
    _ = ctx
