# trt/runner/vision/grid.py

from __future__ import annotations

import pathlib

import torch

from trt.compile import save_trt_engine_module
from trt.modules.export.vision import GridVisionExportModule
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.runner.base import StageContext, StageResult, StageRunner
from trt.utils import free_cuda_memory
from trt.vision import VisionEngineSpec, nchw_to_hwc, vit_visual_edge_config

class GridVisionExportRunner(StageRunner):
    """Stage: NCHW pixels -> visual.engine

    Uses GridVisionExportModule internally.
    Model-specific setup lives in build_*_vision_export_params (spec builder).
    """

    def __init__(self, *, build_spec_fn):
        # build_spec_fn(ctx) -> VisionEngineSpec
        self.build_spec_fn = build_spec_fn

    def run(self, ctx: StageContext) -> StageResult:
        vis_params = self.build_spec_fn(ctx)
        pixel_values = ctx.model_inputs["pixel_values"]

        pixel_values_nchw = pixel_values.to(
            device=ctx.device, dtype=vis_params.input_dtype
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

        with torch.no_grad():
            eager_output = module(images_hwc)

        # mutate spec with probed shapes (for downstream language stage)
        config_seq_len = int(vis_params.config_seq_len or module.output_seq_len)
        vis_params.config_seq_len = config_seq_len
        vis_params.output_hidden_size = int(module.output_hidden_size)
        vis_params.image_embed_flat_shape = (
            int(module.output_num_tokens),
            int(module.output_hidden_size),
        )
        vis_params.image_embed_shape = (
            int(module.batch_size),
            int(module.output_seq_len),
            int(module.output_hidden_size),
        )

        engine_dir = ctx.engine_root / "visual"
        patched = patch_vision_attention(
            vis_params.patch_vision_model,
            batch_size=vis_params.patch_batch_size,
            seq_len=vis_params.patch_seq_len,
            name=vis_params.patch_name,
            allow_attention_mask=vis_params.allow_attention_mask,
        )
        try:
            engine_path = save_trt_engine_module(
                module,
                (images_hwc,),
                engine_dir,
                engine_file="visual.engine",
                model_type=vis_params.model_type,
                component="vision",
                input_names=list(vis_params.io.input_names),
                output_names=list(vis_params.io.output_names),
                example_output=eager_output,
                extra_config=vit_visual_edge_config(
                    vocab_size=vis_params.vocab_size,
                    image_token_id=vis_params.image_token_id,
                    seq_len=config_seq_len,
                ),
                trt_settings=vis_params.trt_settings,
            )
        finally:
            restore_attention(patched)

        return StageResult(
            engine_path=engine_path,
            spec=vis_params,
            tensors={"image_embs": eager_output},
            metadata={"config_seq_len": config_seq_len},
        )