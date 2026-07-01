# trt/executor/models/molmo2/export/vision.py

from trt.hooks.export.plan import ExportPlan
from trt.modules.export.vision import TokenPoolingExportModule


def plan_export(ctx, stage_inputs: dict) -> ExportPlan:
    backbone = ctx.model.model
    media, pooling = backbone.merge_visual_inputs(
        stage_inputs["images"],
        stage_inputs["image_token_pooling"],
    )
    encoder = clone_hf_module_for_export(ctx.profile.vision, device=ctx.device)

    module = TokenPoolingExportModule(
        encoder=encoder,
        sample_media=media,
        sample_pooling_indices=pooling,
    ).eval().to(ctx.device)

    return ExportPlan(
        module=module,
        sample_inputs=(media, pooling),
        input_names=("media", "pooling_indices"),
        output_names=("image_embeddings",),
        engine_dir=ctx.engine_root / "visual",
        engine_file="visual.engine",
        args={
            "encoder": encoder,
            "tensor_aliases": {"image_embeddings": "image_embs"},
        },
    )
