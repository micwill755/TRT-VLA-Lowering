from __future__ import annotations

from trt.context import EdgeContext
from trt.export.mode import ExportMode
from trt.export.pipeline import VLAExportPipeline as LegacyExportPipeline


def run_legacy_export(ctx: EdgeContext, *, mode: ExportMode) -> dict:
    tokenizer = ctx.profile.get_tokenizer(policy=ctx.policy, args=ctx.args)
    hooks = ctx.profile.make_export_hooks(tokenizer=tokenizer, args=ctx.args)
    engine_root = str(ctx.engine_root) if mode is ExportMode.SERIALIZED else None
    result = LegacyExportPipeline(hooks, io=ctx.profile.io).run(
        ctx.model,
        ctx.policy,
        ctx.device,
        ctx.model_inputs,
        mode=mode,
        engine_root=engine_root,
        seed=42,
        max_seq_len=getattr(ctx.args, "max_seq_len", None),
        accuracy_check=True,
    )
    return dict(result.handles)
