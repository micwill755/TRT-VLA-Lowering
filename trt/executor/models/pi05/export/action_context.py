from __future__ import annotations

import torch

from trt.compile import save_trt_engine_module
from trt.context import EdgeContext
from trt.modules.export.language import ContextProjectionExportModule


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    device, dtype = ctx.device, ctx.dtype

    # upstream language output: [B, S, H_lm]
    lm_hidden = inputs["tensors"]["lm_hidden"]
    lm_hidden = lm_hidden.to(device=device, dtype=dtype).contiguous()

    batch_size = int(lm_hidden.shape[0])
    max_seq_len = int(lm_hidden.shape[1])
    hidden_size = int(lm_hidden.shape[2])

    vlln = ctx.model.action_head.vlln
    if getattr(vlln, "weight", None) is not None:
        context_hidden_size = int(vlln.weight.shape[0])
    else:
        context_hidden_size = int(
            ctx.model.backbone.eagle_model.language_model.config.hidden_size
        )

    # same module as inference/action_context.py
    context_module = ContextProjectionExportModule(
        ctx.model.backbone.eagle_linear,
        ctx.model.action_head.vlln,
        ctx.model.action_head.vl_self_attention,
    ).eval().to(device=device, dtype=dtype)

    return {
        "lm_hidden": lm_hidden,
        "context_module": context_module,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "hidden_size": hidden_size,
        "context_hidden_size": context_hidden_size,
        "language_inputs": inputs.get("metadata", {}).get("language_inputs"),
    }


def export(ctx: EdgeContext, inputs: dict) -> dict:
    context_module = inputs["context_module"]
    lm_hidden = inputs["lm_hidden"]

    engine_path = save_trt_engine_module(
        context_module,
        (lm_hidden,),
        ctx.engine_root / "action_context",
        engine_file="context.engine",
        model_type="action_context",
        component="context",
        input_names=["lm_hidden_states"],
        output_names=["vl_embs"],
        extra_config={
            "engine_role": "action_context",
            "batch_size": int(inputs["batch_size"]),
            "max_seq_len": int(inputs["max_seq_len"]),
            "hidden_size": int(inputs["hidden_size"]),
            "context_hidden_size": int(inputs["context_hidden_size"]),
        },
        trt_settings={
            **ctx.trt_settings,
            "offload_module_to_cpu": True,
        },
    )

    # Dummy context embeddings for the downstream diffusion trace — export only
    # needs the [B, S, H_ctx] shape/dtype, not real values (no forward pass here).
    context_embs = torch.zeros(
        inputs["batch_size"],
        inputs["max_seq_len"],
        inputs["context_hidden_size"],
        device=ctx.device,
        dtype=ctx.dtype,
    )

    return {
        "engine_path": engine_path,
        "tensors": {
            "context_embs": context_embs,
        },
        "metadata": {
            "batch_size": inputs["batch_size"],
            "language_inputs": inputs.get("language_inputs"),
            "context_seq_len": inputs["max_seq_len"],
            "context_hidden_size": inputs["context_hidden_size"],
        },
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    return result