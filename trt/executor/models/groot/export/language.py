from __future__ import annotations

from pathlib import Path

import torch

from trt.hooks.export.language_plan import LanguageExportPlan
from trt.io_spec import GROOT_EDGE_IO
from trt.language import (
    compute_vit_expanded_seq_len,
    language_edge_trt_settings,
    language_head_dim,
    make_language_edge_flat_tensors,
)
from trt.executor.models.groot.export.language_helpers import (
    build_language_chat_template,
    clone_language_subgraph,
    pack_language_export_inputs,
)
from trt.modules.export.language import CausalLMExportModule
from trt.runner.base import StageContext
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm


def plan_export(ctx: StageContext, stage_inputs: dict) -> LanguageExportPlan:
    input_ids = stage_inputs["input_ids"]
    attention_mask = stage_inputs["attention_mask"]
    image_embs = stage_inputs["image_embs"]
    image_token_id = int(stage_inputs["image_token_id"])
    seq_len_per_image = int(stage_inputs["seq_len_per_image"])
    dtype = torch.float16

    language_inputs = pack_language_export_inputs(
        ctx.model,
        image_embs=image_embs,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    cloned = clone_language_subgraph(ctx.model, ctx.device, dtype=dtype)
    decoder = cloned.decoder
    cfg = cloned.config
    head_dim = language_head_dim(cfg)
    max_seq_len = compute_vit_expanded_seq_len(
        input_ids,
        image_token_id,
        seq_len_per_image,
    )
    batch_size = int(input_ids.shape[0])

    trace_embeds = language_inputs["inputs_embeds"].to(
        device=ctx.device,
        dtype=dtype,
    ).contiguous()

    wrapper = CausalLMExportModule(decoder, cloned.lm_head, select_layer=-1).eval().to(ctx.device)
    sample_inputs, _ = make_language_edge_flat_tensors(
        trace_embeds,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=head_dim,
        device=ctx.device,
        dtype=dtype,
        static_prefill_seq_len=False,
    )

    return LanguageExportPlan(
        module=wrapper,
        sample_inputs=sample_inputs,
        engine_dir=ctx.engine_root / "language",
        engine_file="language.engine",
        input_names=tuple(GROOT_EDGE_IO.language_input_names(len(decoder.layers))),
        output_names=tuple(GROOT_EDGE_IO.language.output_names),
        decoder=decoder,
        language_model=cloned.language_model,
        language_inputs=language_inputs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hidden_size=int(cfg.hidden_size),
        num_layers=len(decoder.layers),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=head_dim,
        image_token_id=image_token_id,
        seq_len_per_image=seq_len_per_image,
        export_dtype=dtype,
        io=GROOT_EDGE_IO.language,
        trt_settings=language_edge_trt_settings(),
        cleanup_modules=(wrapper, cloned.language_model),
        model_type="language",
        component="language",
    )


def save_artifacts(ctx: StageContext, plan: LanguageExportPlan, engine_path: Path) -> None:
    del engine_path
    export_state = getattr(ctx, "export_state", {})
    tokenizer = export_state.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("preprocess must stash tokenizer on ctx.export_state['tokenizer']")

    save_embedding_table(plan.language_model, plan.engine_dir)
    save_tokenizer_for_edge_llm(
        plan.engine_dir,
        tokenizer=tokenizer,
        chat_template=build_language_chat_template(tokenizer),
    )


def metadata(ctx: StageContext, plan: LanguageExportPlan, output) -> dict:
    del ctx
    if isinstance(output, (tuple, list)):
        lm_hidden_states = output[1]
    else:
        lm_hidden_states = output

    return {
        "lm_hidden_states": lm_hidden_states,
        "language_inputs": plan.language_inputs,
        "max_seq_len": plan.max_seq_len,
        "hidden_size": plan.hidden_size,
        "image_token_id": plan.image_token_id,
        "seq_len_per_image": plan.seq_len_per_image,
    }
