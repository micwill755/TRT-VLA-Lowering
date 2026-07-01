from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.export.settings import VISION_TRT_SETTINGS
from trt.hooks.export.plan import ExportPlan
from trt.modules.export.vision import GridVisionExportModule
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.runner.base import StageContext
from trt.vision import nchw_to_hwc, vit_visual_edge_config
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


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
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

    return ExportPlan(
        module=module,
        sample_inputs=(images_hwc,),
        input_names=tuple(vis_params.io.input_names),
        output_names=tuple(vis_params.io.output_names),
        engine_dir=ctx.engine_root / "visual",
        engine_file="visual.engine",
        model_type="visual",
        component="vision",
        trt_settings=vis_params.trt_settings,
        cleanup_modules=(module, vis_params.visual_vision_model),
        args={
            "patch_target": vis_params.patch_vision_model,
            "patch_batch_size": vis_params.patch_batch_size,
            "patch_seq_len": vis_params.patch_seq_len,
            "patch_name": vis_params.patch_name,
            "allow_attention_mask": vis_params.allow_attention_mask,
            "vocab_size": vis_params.vocab_size,
            "image_token_id": vis_params.image_token_id,
            "config_seq_len": config_seq_len,
            "tensor_aliases": {"image_embeddings": "image_embs"},
        },
    )


def compile(plan: ExportPlan, eager_output) -> Path:
    args = plan.args
    patched = patch_vision_attention(
        args["patch_target"],
        batch_size=args["patch_batch_size"],
        seq_len=args["patch_seq_len"],
        name=args["patch_name"],
        allow_attention_mask=args["allow_attention_mask"],
    )
    try:
        return save_trt_engine_module(
            plan.module,
            plan.sample_inputs,
            plan.engine_dir,
            engine_file=plan.engine_file,
            model_type=plan.model_type or "visual",
            component=plan.component or "vision",
            input_names=list(plan.input_names),
            output_names=list(plan.output_names),
            example_output=eager_output,
            extra_config=vit_visual_edge_config(
                vocab_size=args["vocab_size"],
                image_token_id=args["image_token_id"],
                seq_len=args["config_seq_len"],
            ),
            trt_settings=plan.trt_settings,
        )
    finally:
        restore_attention(patched)


def metadata(ctx: StageContext, plan: ExportPlan, output) -> dict:
    del ctx, output
    args = plan.args
    return {
        "config_seq_len": args["config_seq_len"],
        "image_token_id": args["image_token_id"],
        "output_hidden_size": int(plan.module.output_hidden_size),
    }
