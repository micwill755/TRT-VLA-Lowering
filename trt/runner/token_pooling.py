# trt/runner/vision/token_pooling.py

from __future__ import annotations

import torch

from trt.compile import save_trt_engine_module
from trt.modules.export.vision import TokenPoolingExportModule
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.runner.base import StageContext, StageResult, StageRunner
from trt.utils import free_cuda_memory


class TokenPoolingVisionExportRunner(StageRunner):
    """Stage: (media, pooling_indices) -> visual.engine

    Preprocess (merge_visual_inputs) happens here, not in ExportModule.
    """

    def __init__(self, *, build_spec_fn):
        # build_spec_fn(ctx) -> TokenPoolingVisionEngineSpec
        self.build_spec_fn = build_spec_fn

    def run(self, ctx: StageContext) -> StageResult:
        spec = self.build_spec_fn(ctx)

        media, pooling_indices = spec.prepare_inputs(ctx.model_inputs, ctx.device)
        # e.g. backbone.merge_visual_inputs(...) inside prepare_inputs

        module = TokenPoolingExportModule(
            encoder=spec.encoder,
            sample_media=media,
            sample_pooling_indices=pooling_indices,
            cast_output_to_input_dtype=spec.cast_output_to_input_dtype,
        ).eval().to(ctx.device)

        with torch.no_grad():
            eager_output = module(media, pooling_indices)

        spec.output_num_tokens = int(module.output_num_tokens)
        spec.output_hidden_size = int(module.output_hidden_size)

        engine_dir = ctx.engine_root / "visual"
        patched = patch_vision_attention(
            spec.patch_vision_model,
            batch_size=spec.patch_batch_size,
            seq_len=spec.patch_seq_len,
            name=spec.patch_name,
            allow_attention_mask=spec.allow_attention_mask,
        )
        try:
            engine_path = save_trt_engine_module(
                module,
                (media, pooling_indices),
                engine_dir,
                engine_file="visual.engine",
                model_type=spec.model_type,
                component="vision",
                input_names=list(spec.io.input_names),    # ("media", "pooling_indices")
                output_names=list(spec.io.output_names),
                example_output=eager_output,
                extra_config=spec.edge_config(),
                trt_settings=spec.trt_settings,
            )
        finally:
            restore_attention(patched)

        return StageResult(
            engine_path=engine_path,
            spec=spec,
            tensors={"image_embs": eager_output},
        )