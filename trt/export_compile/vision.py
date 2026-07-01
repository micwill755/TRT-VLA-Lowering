from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.vision_plan import VisionExportPlan
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.vision import vit_visual_edge_config


def compile_vision_plan(plan: VisionExportPlan) -> Path:
    with torch.no_grad():
        eager_output = plan.module(*plan.sample_inputs)

    patched = patch_vision_attention(
        plan.patch_target,
        batch_size=plan.patch_batch_size,
        seq_len=plan.patch_seq_len,
        name=plan.patch_name,
        allow_attention_mask=plan.allow_attention_mask,
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
                vocab_size=plan.vocab_size,
                image_token_id=plan.image_token_id,
                seq_len=plan.config_seq_len,
            ),
            trt_settings=plan.trt_settings,
        )
    finally:
        restore_attention(patched)
