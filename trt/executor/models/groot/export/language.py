"""GR00T language export hooks: Eagle LM prefill → TRT LLMEngineRunner engine.

Stage 1 in the GROOT export pipeline (after vision). ``glue.vision_to_language`` wires
stage-0 metadata into ``stage_inputs``; ``plan_export`` builds an ``ExportPlan``;
``compile`` traces ``CausalLMExportModule`` and writes ``language/language.engine``;
``save_artifacts`` writes embedding table + tokenizer JSON for C++ runtime.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.compile import save_trt_engine_module
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
from trt.plugin.plugin_utils import patch_language_attention, restore_attention
from trt.rope import make_rope_rotary_cos_sin
from trt.runner.base import StageContext
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm
from trt.utils import clone_hf_module_for_export


@torch.no_grad()
def _pack_language_export_inputs(
    model: nn.Module,
    *,
    image_embs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build LM prefill ``inputs_embeds`` by splicing vision rows into token embeds.

    Mirrors eager GROOT: text tokens are embedded, then rows at ``image_token_id``
    positions are overwritten with vision outputs (during export, ``image_embs`` from
    glue are zeros shaped for trace only).

    Shapes::

        input_ids, attention_mask     [B, T]     chat tokens (T = prompt length)
        image_embs                    [N_slots, H_lm]  one row per image placeholder
        input_embs (after embed)      [B, T, H_lm]
        inputs_embeds (returned)      [B, T, H_lm]  vision rows spliced in
        position_ids                  [B, T]
        image_token_mask              [B, T]     bool mask where id == image_token_id

    ``N_slots`` = count of ``image_token_id`` in ``input_ids`` (matches glue dummy
    ``image_embs`` row count from stage 0 ``config_seq_len`` metadata).
    """
    eagle = model.backbone.eagle_model
    image_token_id = int(getattr(eagle, "image_token_index", eagle.config.image_token_index))
    token_embed_fn = eagle.language_model.get_input_embeddings()

    input_embs = token_embed_fn(input_ids)  # [B, T, H_lm]
    attention_mask = attention_mask.to(device=input_embs.device, dtype=torch.long)

    bsz, seq_len, hidden = input_embs.shape
    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)

    image_token_mask = flat_ids == image_token_id
    flat_image_embs = image_embs.reshape(-1, hidden).to(
        device=flat_embs.device,
        dtype=flat_embs.dtype,
    )

    num_slots = int(image_token_mask.sum().item())
    if flat_image_embs.shape[0] < num_slots:
        raise ValueError(
            f"Not enough image embeddings: {flat_image_embs.shape[0]} for {num_slots} slots"
        )

    flat_embs[image_token_mask] = flat_image_embs[:num_slots]
    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden)
    pad_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

    return {
        "inputs_embeds": inputs_embeds,
        "pad_mask": pad_mask,
        "attention_mask": attention_mask,
        "position_ids": torch.cumsum(pad_mask, dim=1) - 1,
        "image_token_mask": image_token_mask.reshape(bsz, seq_len),
    }


def _build_language_chat_template(tokenizer: Any) -> dict[str, Any]:
    """Build C++ ``llm_inference`` chat-template JSON from the Eagle tokenizer.

    Probes ``apply_chat_template`` with placeholder strings to split role prefixes /
    suffixes and the generation prompt. Written beside ``language.engine`` in
    ``save_artifacts`` — not used during TRT trace.
    """
    im_end = tokenizer.eos_token
    if not im_end:
        raise ValueError("Tokenizer eos_token required")

    system_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": "SYS"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TEXTONLY"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    with_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TEXTONLY"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assistant_only = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "TEXTONLY"},
            {"role": "assistant", "content": "ASSIST"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    system_prefix = system_only.split("SYS", 1)[0]
    system_suffix = "SYS" + system_only.split("SYS", 1)[1]
    user_prefix = user_only.split("TEXTONLY", 1)[0]
    user_suffix = "TEXTONLY" + user_only.split("TEXTONLY", 1)[1]
    assistant_prefix = assistant_only[len(user_only) :].split("ASSIST", 1)[0]
    assistant_suffix = "ASSIST" + assistant_only.split("ASSIST", 1)[1]
    generation_prompt = with_gen[len(user_only) :]

    return {
        "model_path": "groot-vitrunner",
        "roles": {
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "assistant": {"prefix": assistant_prefix, "suffix": assistant_suffix},
        },
        "content_types": {"image": {"format": "<img><IMG_CONTEXT></img>"}},
        "generation_prompt": generation_prompt,
        "default_system_prompt": "You are a helpful assistant.",
    }


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
    """Build the GR00T language TRT export plan (Eagle decoder prefill + lm_head).

    ``stage_inputs`` come from ``glue.vision_to_language`` (upstream stage 0).

    Shape flow (typical libero prompt, B=1)::

        input_ids, attention_mask   [B, T]           tokenized chat (pre-expansion)
        image_embs                  [N_slots, H_lm]  dummy zeros for trace
        inputs_embeds               [B, T, H_lm]     after _pack_language_export_inputs
        max_seq_len                 scalar T_exp     prompt len after VitRunner expands
                                                     each image placeholder → S_out rows
        trace sample_inputs:
          inputs_embeds             [B, T_exp, H_lm]
          rope_rotary_cos_sin       [T_exp, ...]  (model-specific layout)
          context_lengths           [B]
          kvcache_start_index       [0]
          last_token_ids            [B, 1]
          past_key_values_i         [B, 2, KvH, T_exp, head_dim] × num_layers
        module outputs (Edge-LLM):
          logits, lm_hidden_states, prefix_k, prefix_v

    ``T_exp`` = ``compute_vit_expanded_seq_len`` — larger than ``T`` when image
    placeholders are expanded to ``seq_len_per_image`` (from vision ``config_seq_len``).
    """
    # --- 1. Stage inputs from glue (vision_to_language) -------------------
    input_ids = stage_inputs["input_ids"]              # [B, T]
    attention_mask = stage_inputs["attention_mask"]    # [B, T]
    image_embs = stage_inputs["image_embs"]            # [N_slots, H_lm]
    image_token_id = int(stage_inputs["image_token_id"])
    seq_len_per_image = int(stage_inputs["seq_len_per_image"])  # S_out from vision
    dtype = torch.float16

    # --- 2. Prefill embedding table (text + spliced vision rows) -----------
    language_inputs = _pack_language_export_inputs(
        ctx.model,
        image_embs=image_embs,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # --- 3. Clone Eagle LM subgraph (see clone_hf_module_for_export) -------
    # Trim config to active decoder depth; clone LM + lm_head for trace/compile.
    source_lm = ctx.model.backbone.eagle_model.language_model
    source_decoder = getattr(source_lm, "model", source_lm)
    lm_config = copy.deepcopy(source_lm.config)
    lm_config.num_hidden_layers = len(source_decoder.layers)
    language_model = clone_hf_module_for_export(
        source_lm,
        ctx.device,
        dtype=dtype,
        config=lm_config,
    )
    decoder = getattr(language_model, "model", language_model)
    lm_head = clone_hf_module_for_export(language_model.lm_head, ctx.device, dtype=dtype)

    cfg = language_model.config
    head_dim = language_head_dim(cfg)

    # --- 4. Static sequence length for Edge-LLM engine profiles -----------
    # T_exp: length after C++ replaces each image placeholder with S_out vision rows.
    max_seq_len = compute_vit_expanded_seq_len(
        input_ids,
        image_token_id,
        seq_len_per_image,
    )
    batch_size = int(input_ids.shape[0])  # B

    trace_embeds = language_inputs["inputs_embeds"].to(
        device=ctx.device,
        dtype=dtype,
    ).contiguous()  # [B, T, H_lm] — sliced to T_exp in make_language_edge_flat_tensors

    # --- 5. Trace target: decoder loop + lm_head (PluginAttention inside) -
    exportModule = CausalLMExportModule(decoder, lm_head, select_layer=-1).eval().to(ctx.device)

    # --- 6. Flat Edge-LLM bindings for torch.export / TRT -----------------
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        max_seq_len,
        ctx.device,
        language_model=language_model,
        position_ids=language_inputs.get("position_ids"),
    )
    sample_inputs, _ = make_language_edge_flat_tensors(
        trace_embeds,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=head_dim,
        device=ctx.device,
        dtype=dtype,
        rope_rotary_cos_sin=rope_rotary_cos_sin,
        static_prefill_seq_len=True,  # GR00T: prefix KV must match full prompt
    )

    # --- 7. ExportPlan — consumed by ExportRunner + compile() hook --------
    return ExportPlan(
        module=exportModule,
        sample_inputs=sample_inputs,
        input_names=tuple(GROOT_EDGE_IO.language_input_names(len(decoder.layers))),
        output_names=tuple(GROOT_EDGE_IO.language.output_names),
        engine_dir=ctx.engine_root / "language",
        engine_file="language.engine",
        model_type="language",
        component="language",
        trt_settings=language_edge_trt_settings(),
        cleanup_modules=(exportModule, language_model),
        args={
            "decoder": decoder,
            "language_model": language_model,
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
            "static_prefill_seq_len": True,
            "enable_bidirectional_prefill": 0,
            "context_hidden_size": None,
            "tensor_aliases": {"lm_hidden_states": "hidden_states"},
        },
    )


def compile(plan: ExportPlan) -> Path:
    """Trace ``plan.module`` and write ``language/language.engine`` + config JSON.

    Runs on the **cloned** LM from ``plan_export``. Steps:

    1. ``make_language_edge_input_specs`` — multi-profile TRT shapes (prefill/decode).
    2. ``patch_language_attention`` — swap decoder layers to ``PluginAttention``.
    3. ``save_trt_engine_module`` — torch.export + TensorRT compile of prefill graph.
       Flat inputs: ``inputs_embeds [B,T_exp,H]``, RoPE table, context metadata,
       ``past_key_values_* [B,2,KvH,T_exp,head_dim]`` per layer.
    4. ``language_edge_llm_config`` — C++ ``LLMEngineRunner`` metadata (max seq,
       layer count, image_token_id, etc.).
    5. ``restore_attention`` in ``finally`` before ``cleanup_modules`` deletes clones.
    """
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
            example_output=None,
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
    """Write embedding table + tokenizer/chat JSON beside ``language.engine``.

    Required by C++ Edge-LLM at load time (not part of the TRT graph). Uses the
    cloned ``language_model`` weights and ``ctx.export_state['tokenizer']`` from
    preprocess. Runs before ``cleanup_modules`` frees the clone.
    """
    export_state = getattr(ctx, "export_state", {})
    tokenizer = export_state.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("preprocess must stash tokenizer on ctx.export_state['tokenizer']")

    language_model = plan.args["language_model"]
    save_embedding_table(language_model, plan.engine_dir)
    save_tokenizer_for_edge_llm(
        plan.engine_dir,
        tokenizer=tokenizer,
        chat_template=_build_language_chat_template(tokenizer),
    )


def metadata(ctx: StageContext, plan: ExportPlan) -> dict:
    """Return stage-1 artifacts stored on ``StageResult.metadata``.

    ``language_to_action_context`` reads ``hidden_size`` / ``max_seq_len`` to build
    dummy tensors for action-context export. ``language_inputs`` preserves the packed
    prefill embeds for benchmark parity.
    """
    args = plan.args
    return {
        "batch_size": args["batch_size"],
        "language_inputs": args["language_inputs"],
        "max_seq_len": args["max_seq_len"],           # T_exp
        "hidden_size": args["hidden_size"],           # H_lm
        "image_token_id": args["image_token_id"],
        "seq_len_per_image": args["seq_len_per_image"],  # S_out from vision
    }
