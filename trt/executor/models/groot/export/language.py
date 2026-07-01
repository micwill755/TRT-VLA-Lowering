from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.executor.models.groot.export.language_helpers import (
    build_language_chat_template,
    clone_language_subgraph,
    pack_language_export_inputs,
)
from trt.hooks.export.plan import ExportPlan
from trt.io_spec import GROOT_EDGE_IO
from trt.language import (
    compute_vit_expanded_seq_len,
    language_edge_llm_config,
    language_edge_output_names,
    language_edge_trt_settings,
    language_head_dim,
    make_language_edge_flat_tensors,
    make_language_edge_input_specs,
)
from trt.modules.export.language import CausalLMExportModule
from trt.plugin_utils import patch_language_attention, restore_attention
from trt.runner.base import StageContext
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
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

    return ExportPlan(
        module=wrapper,
        sample_inputs=sample_inputs,
        input_names=tuple(GROOT_EDGE_IO.language_input_names(len(decoder.layers))),
        output_names=tuple(GROOT_EDGE_IO.language.output_names),
        engine_dir=ctx.engine_root / "language",
        engine_file="language.engine",
        model_type="language",
        component="language",
        trt_settings=language_edge_trt_settings(),
        cleanup_modules=(wrapper, cloned.language_model),
        args={
            "decoder": decoder,
            "language_model": cloned.language_model,
            "language_inputs": language_inputs,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "hidden_size": int(cfg.hidden_size),
            "num_layers": len(decoder.layers),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "head_dim": head_dim,
            "image_token_id": image_token_id,
            "seq_len_per_image": seq_len_per_image,
            "static_prefill_seq_len": False,
            "enable_bidirectional_prefill": 0,
            "context_hidden_size": None,
            "tensor_aliases": {"lm_hidden_states": "hidden_states"},
        },
    )


def compile(plan: ExportPlan, eager_output) -> Path:
    args = plan.args
    input_specs = make_language_edge_input_specs(
        list(plan.input_names),
        plan.sample_inputs,
        batch_size=args["batch_size"],
        max_seq_len=args["max_seq_len"],
        static_prefill_seq_len=args["static_prefill_seq_len"],
    )
    output_names = language_edge_output_names(plan.output_names, args["num_layers"])

    patched = patch_language_attention(
        args["decoder"],
        hidden_size=args["hidden_size"],
        num_attention_heads=args["num_attention_heads"],
        num_key_value_heads=args["num_key_value_heads"],
        head_dim=args["head_dim"],
        enable_bidirectional_prefill=args["enable_bidirectional_prefill"],
    )
    try:
        return save_trt_engine_module(
            plan.module,
            plan.sample_inputs,
            plan.engine_dir,
            engine_file=plan.engine_file,
            model_type=plan.model_type or "language",
            component=plan.component or "language",
            input_names=list(plan.input_names),
            output_names=output_names,
            example_output=eager_output,
            extra_config=language_edge_llm_config(
                args["language_model"].config,
                max_seq_len=args["max_seq_len"],
                batch_size=args["batch_size"],
                num_layers=args["num_layers"],
                context_hidden_size=args["context_hidden_size"],
                image_token_id=args["image_token_id"],
            ),
            input_specs=input_specs,
            flat_tensors=plan.sample_inputs,
            trt_settings=plan.trt_settings,
        )
    finally:
        restore_attention(patched)


def save_artifacts(ctx: StageContext, plan: ExportPlan, engine_path: Path) -> None:
    del engine_path
    export_state = getattr(ctx, "export_state", {})
    tokenizer = export_state.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("preprocess must stash tokenizer on ctx.export_state['tokenizer']")

    language_model = plan.args["language_model"]
    save_embedding_table(language_model, plan.engine_dir)
    save_tokenizer_for_edge_llm(
        plan.engine_dir,
        tokenizer=tokenizer,
        chat_template=build_language_chat_template(tokenizer),
    )


def metadata(ctx: StageContext, plan: ExportPlan, output) -> dict:
    del ctx
    args = plan.args
    if isinstance(output, (tuple, list)):
        lm_hidden_states = output[1]
    else:
        lm_hidden_states = output

    return {
        "lm_hidden_states": lm_hidden_states,
        "language_inputs": args["language_inputs"],
        "max_seq_len": args["max_seq_len"],
        "hidden_size": args["hidden_size"],
        "image_token_id": args["image_token_id"],
        "seq_len_per_image": args["seq_len_per_image"],
    }
