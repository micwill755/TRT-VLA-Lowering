from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.language_plan import LanguageExportPlan
from trt.language import (
    language_edge_llm_config,
    language_edge_output_names,
    make_language_edge_input_specs,
)
from trt.plugin_utils import patch_language_attention, restore_attention


def compile_language_plan(plan: LanguageExportPlan) -> Path:
    input_specs = make_language_edge_input_specs(
        list(plan.input_names),
        plan.sample_inputs,
        batch_size=plan.batch_size,
        max_seq_len=plan.max_seq_len,
        static_prefill_seq_len=plan.static_prefill_seq_len,
    )
    output_names = language_edge_output_names(plan.output_names, plan.num_layers)

    patched = patch_language_attention(
        plan.decoder,
        hidden_size=plan.hidden_size,
        num_attention_heads=plan.num_attention_heads,
        num_key_value_heads=plan.num_key_value_heads,
        head_dim=plan.head_dim,
        enable_bidirectional_prefill=plan.enable_bidirectional_prefill,
    )
    try:
        with torch.no_grad():
            example_output = plan.module(*plan.sample_inputs)

        return save_trt_engine_module(
            plan.module,
            plan.sample_inputs,
            plan.engine_dir,
            engine_file=plan.engine_file,
            model_type=plan.model_type or "language",
            component=plan.component or "language",
            input_names=list(plan.input_names),
            output_names=output_names,
            example_output=example_output,
            extra_config=language_edge_llm_config(
                plan.language_model.config,
                max_seq_len=plan.max_seq_len,
                batch_size=plan.batch_size,
                num_layers=plan.num_layers,
                context_hidden_size=plan.context_hidden_size,
                image_token_id=plan.image_token_id,
            ),
            input_specs=input_specs,
            flat_tensors=plan.sample_inputs,
            trt_settings=plan.trt_settings,
        )
    finally:
        restore_attention(patched)
