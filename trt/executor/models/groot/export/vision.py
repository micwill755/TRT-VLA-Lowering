from __future__ import annotations

import torch

from trt.export.settings import VISION_TRT_SETTINGS
from trt.hooks.export.vision_plan import VisionExportPlan
from trt.modules.export.vision import GridVisionExportModule
from trt.runner.base import StageContext
from trt.vision import nchw_to_hwc
from trt.vision_builders import build_groot_vision_export_params


def _pixel_values(ctx, stage_inputs: dict) -> torch.Tensor:
    export_state = getattr(ctx, "export_state", {})
    if "pixel_values" in export_state:
        return export_state["pixel_values"]
    if "pixel_values" in stage_inputs:
        return stage_inputs["pixel_values"]
    tokenized = stage_inputs.get("tokenized_data") or export_state.get("tokenized")
    if tokenized is not None and "pixel_values" in tokenized:
        return tokenized["pixel_values"]
    raise KeyError("pixel_values not found in export_state or stage inputs")


def plan_export(ctx: StageContext, stage_inputs: dict) -> VisionExportPlan:
    pixel_values = _pixel_values(ctx, stage_inputs)

    vis_params = build_groot_vision_export_params(
        ctx.model,
        pixel_values,
        ctx.device,
        trt_settings=VISION_TRT_SETTINGS,
        input_dtype=torch.float16,
    )
    pixel_values_nchw = pixel_values.to(
        device=ctx.device,
        dtype=vis_params.input_dtype,
    ).contiguous()
    images_hwc = nchw_to_hwc(pixel_values_nchw)

    module = GridVisionExportModule(
        vision_model=vis_params.visual_vision_model,
        projector=vis_params.projector,
        sample_pixel_values=images_hwc,
        select_layer=vis_params.select_layer,
        pixel_shuffle=vis_params.pixel_shuffle,
        downsample_ratio=vis_params.downsample_ratio,
        force_float32_input=vis_params.force_float32_input,
        cast_output_to_input_dtype=vis_params.cast_output_to_input_dtype,
        vision_kwargs=vis_params.vision_kwargs,
    ).eval().to(ctx.device)

    config_seq_len = int(vis_params.config_seq_len or module.output_seq_len)

    return VisionExportPlan(
        module=module,
        sample_inputs=(images_hwc,),
        engine_dir=ctx.engine_root / "visual",
        engine_file="visual.engine",
        input_names=tuple(vis_params.io.input_names),
        output_names=tuple(vis_params.io.output_names),
        patch_target=vis_params.patch_vision_model,
        patch_batch_size=vis_params.patch_batch_size,
        patch_seq_len=vis_params.patch_seq_len,
        vocab_size=vis_params.vocab_size,
        image_token_id=vis_params.image_token_id,
        config_seq_len=config_seq_len,
        patch_name=vis_params.patch_name,
        allow_attention_mask=vis_params.allow_attention_mask,
        trt_settings=vis_params.trt_settings,
        cleanup_modules=(module, vis_params.visual_vision_model),
        model_type="visual",
        component="vision",
    )


def metadata(ctx: StageContext, plan: VisionExportPlan, output) -> dict:
    del ctx, output
    return {
        "config_seq_len": plan.config_seq_len,
        "image_token_id": plan.image_token_id,
        "output_hidden_size": int(plan.module.output_hidden_size),
    }
