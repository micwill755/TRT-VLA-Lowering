# trt/executor/models/groot/vision.py

from trt.hooks.export_plan import ExportPlan
from trt.modules.export.vision import GridVisionExportModule
from trt.runner.base import StageContext
from trt.vision import nchw_to_hwc
from trt.vision_builders import clone_groot_vision_modules  # extract from groot.py

def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
    pixel_values = stage_inputs["pixel_values"]
    images_hwc = nchw_to_hwc(pixel_values.to(device=ctx.device, dtype=torch.float16))

    vision_model, projector = clone_groot_vision_modules(ctx.model, device=ctx.device)

    module = GridVisionExportModule(
        vision_model=vision_model,
        projector=projector,
        sample_pixel_values=images_hwc,
        select_layer=...,
        pixel_shuffle=...,
        downsample_ratio=...,
    ).eval().to(ctx.device)

    return ExportPlan(
        module=module,
        sample_inputs=(images_hwc,),
        engine_dir=ctx.engine_root / "visual",
        engine_file="visual.engine",
        input_names=("pixel_values",),
        output_names=("image_embeddings",),
        extra_config={...},
        trt_settings=VISION_TRT_SETTINGS,
        patch_target=vision_model.vision_model,
        patch_batch_size=...,
        patch_seq_len=...,
        cleanup_modules=(module, vision_model, projector),
    )

def metadata(ctx: StageContext, plan: ExportPlan, output) -> dict:
    m = plan.module
    return {
        "config_seq_len": m.output_seq_len,
        "image_token_id": ctx.profile.image_token_id,
        "output_hidden_size": m.output_hidden_size,
    }