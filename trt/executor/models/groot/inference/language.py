from __future__ import annotations

import torch

from trt.context import EdgeContext, LanguageOutputs
from trt.language import language_head_dim, make_language_edge_flat_tensors
from trt.packing import pack_groot_language_inputs
from trt.rope import make_rope_rotary_cos_sin
from trt.runner.inference import InferenceStageResult


def _prepare_language(ctx: EdgeContext) -> None:
    infer = ctx.inference
    infer.language_inputs = pack_groot_language_inputs(
        ctx.model,
        infer.image_embs,
        infer.tokenized["input_ids"],
        infer.tokenized["attention_mask"],
    )


def _build_language_prefill_inputs(
    language_inputs: dict,
    *,
    language_model,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    prefix_embs = language_inputs["inputs_embeds"]
    batch_size = int(prefix_embs.shape[0])
    seq_len = int(prefix_embs.shape[1])
    max_seq_len = max(int(max_seq_len), seq_len)
    dtype = prefix_embs.dtype

    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        language_model.config,
        max_seq_len=max_seq_len,
        device=device,
        language_model=language_model,
    )

    flat_tensors, _ = make_language_edge_flat_tensors(
        prefix_embs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=int(num_layers),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim or language_head_dim(language_model.config)),
        device=device,
        dtype=dtype,
        rope_rotary_cos_sin=rope_rotary_cos_sin,
        static_prefill_seq_len=True,
    )
    return flat_tensors


def _run_trt_language(ctx: EdgeContext, language_module) -> None:
    language_model = ctx.model.backbone.eagle_model.language_model
    decoder = getattr(language_model, "model", language_model)
    seq_len = int(ctx.inference.language_inputs["inputs_embeds"].shape[1])
    max_seq_len = seq_len
    if language_module is not None:
        max_seq_len = int(
            getattr(language_module, "max_seq_len", language_module.engine.config["max_seq_len"])
        )
    prefill = _build_language_prefill_inputs(
        ctx.inference.language_inputs,
        language_model=language_model,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(language_model.config.num_key_value_heads),
        head_dim=int(language_head_dim(language_model.config)),
        max_seq_len=max(max_seq_len, seq_len),
        device=ctx.device,
    )
    outputs = language_module(*prefill)
    if isinstance(outputs, LanguageOutputs):
        ctx.inference.lm = outputs
    elif isinstance(outputs, tuple):
        logits, lm_hidden = outputs
        del logits
        ctx.inference.lm = LanguageOutputs(lm_hidden_states=lm_hidden)
    else:
        ctx.inference.lm = LanguageOutputs(lm_hidden_states=outputs)


def run_eager(ctx: EdgeContext) -> InferenceStageResult:
    _prepare_language(ctx)
    eagle = ctx.model.backbone.eagle_model
    language_model = eagle.language_model
    language_inputs = ctx.inference.language_inputs
    inputs_embeds = language_inputs["inputs_embeds"]
    attention_mask = language_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = language_inputs["pad_mask"].to(dtype=torch.long)

    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    select_layer = int(getattr(eagle, "select_layer", -1))
    if select_layer == -1:
        lm_hidden = outputs.last_hidden_state
    else:
        lm_hidden = outputs.hidden_states[select_layer]

    ctx.inference.lm = LanguageOutputs(lm_hidden_states=lm_hidden)
    return InferenceStageResult(
        tensors={"lm_hidden_states": lm_hidden},
        metadata={"language_inputs": dict(ctx.inference.language_inputs)},
    )


def run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    _prepare_language(ctx)
    module = ctx.handles.serialized.language
    if module is None:
        raise RuntimeError("serialized TRT backend missing language module")
    _run_trt_language(ctx, module)
    lm_hidden = ctx.inference.lm.lm_hidden_states
    if lm_hidden is None:
        raise RuntimeError("language stage did not produce lm_hidden_states")
    return InferenceStageResult(
        tensors={"lm_hidden_states": lm_hidden},
        metadata={"language_inputs": dict(ctx.inference.language_inputs)},
    )


def run_trt(ctx: EdgeContext) -> InferenceStageResult:
    _prepare_language(ctx)
    module = ctx.handles.in_memory.language
    if module is None:
        raise RuntimeError("in-memory TRT backend missing language module")
    _run_trt_language(ctx, module)
    lm_hidden = ctx.inference.lm.lm_hidden_states
    if lm_hidden is None:
        raise RuntimeError("language stage did not produce lm_hidden_states")
    return InferenceStageResult(
        tensors={"lm_hidden_states": lm_hidden},
        metadata={"language_inputs": dict(ctx.inference.language_inputs)},
    )
