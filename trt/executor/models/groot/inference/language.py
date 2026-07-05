from __future__ import annotations

import torch

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.language import language_head_dim, make_language_edge_flat_tensors, unpack_vla_prefix_language_outputs
from trt.modules.export.language import gather_last_token_hidden
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
        position_ids=language_inputs.get("position_ids"),
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


def _language_prefill_bundle(
    ctx: EdgeContext,
    language_module,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
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
    return language_model, prefill


@torch.no_grad()
def _eager_last_token_logits(
    language_model,
    language_inputs: dict,
    last_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Last-token logits from HF eager forward, gathered at ``last_token_ids``."""
    inputs_embeds = language_inputs["inputs_embeds"]
    attention_mask = language_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = language_inputs["pad_mask"].to(dtype=torch.long)

    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits
    if logits.ndim == 3:
        return gather_last_token_hidden(logits, last_token_ids).float()
    return logits.float()


@torch.no_grad()
def _trt_last_token_logits(language_module, prefill: tuple[torch.Tensor, ...]) -> torch.Tensor:
    outputs = language_module(*prefill)
    if isinstance(outputs, tuple):
        return outputs[0].float()
    return outputs.float()


@torch.no_grad()
def compare_language_logits(
    ctx: EdgeContext,
    language_module=None,
    *,
    print_metrics: bool = True,
) -> dict[str, float]:
    """Compare last-token language logits: serialized TRT engine vs eager HF.

    Requires ``ctx.inference.image_embs`` (run vision first or set manually).
    Uses the same packed ``inputs_embeds`` and Edge-LLM prefill bindings as
    serialized mode, including ``last_token_ids`` from ``make_language_edge_flat_tensors``.
    """
    if ctx.inference.image_embs is None:
        raise RuntimeError("compare_language_logits requires ctx.inference.image_embs")

    if language_module is None:
        language_module = ctx.handles.serialized.language
    if language_module is None:
        raise RuntimeError("compare_language_logits requires a loaded language engine")

    _prepare_language(ctx)
    language_model, prefill = _language_prefill_bundle(ctx, language_module)
    last_token_ids = prefill[4]

    trt_logits = _trt_last_token_logits(language_module, prefill)
    eager_logits = _eager_last_token_logits(
        language_model,
        ctx.inference.language_inputs,
        last_token_ids,
    )

    from trt.measure import tensor_error_metrics, tensor_parity_metrics

    if print_metrics:
        tensor_error_metrics("language logits", trt_logits, eager_logits)
    return tensor_parity_metrics(trt_logits, eager_logits)


def _last_token_ids(ctx: EdgeContext) -> torch.Tensor:
    _, prefill = _language_prefill_bundle(ctx, language_module=None)
    return prefill[4]


def _store_language_tensors(
    ctx: EdgeContext,
    *,
    lm_hidden_states: torch.Tensor,
    logits: torch.Tensor,
    prefix_k: torch.Tensor | None = None,
    prefix_v: torch.Tensor | None = None,
) -> None:
    ctx.inference.lm_hidden_states = lm_hidden_states
    ctx.inference.logits = logits
    ctx.inference.prefix_k = prefix_k
    ctx.inference.prefix_v = prefix_v


def _language_stage_result(
    ctx: EdgeContext,
    *,
    lm_hidden_states: torch.Tensor,
    logits: torch.Tensor,
    prefix_k: torch.Tensor | None = None,
    prefix_v: torch.Tensor | None = None,
) -> InferenceStageResult:
    _store_language_tensors(
        ctx,
        lm_hidden_states=lm_hidden_states,
        logits=logits,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
    )
    tensors = {
        "lm_hidden_states": lm_hidden_states,
        "logits": logits,
    }
    if prefix_k is not None:
        tensors["prefix_k"] = prefix_k
    if prefix_v is not None:
        tensors["prefix_v"] = prefix_v
    return InferenceStageResult(
        tensors=tensors,
        metadata={"language_inputs": dict(ctx.inference.language_inputs)},
    )


def _run_trt_language(
    ctx: EdgeContext,
    language_module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    _, prefill = _language_prefill_bundle(ctx, language_module)
    outputs = language_module(*prefill)
    if isinstance(outputs, tuple):
        if len(outputs) == 4:
            logits, lm_hidden, prefix_k, prefix_v = unpack_vla_prefix_language_outputs(outputs)
            return lm_hidden, logits.float(), prefix_k, prefix_v
        if len(outputs) == 2:
            logits, lm_hidden = outputs
            return lm_hidden, logits.float(), None, None
        raise RuntimeError(f"unexpected language TRT output tuple length: {len(outputs)}")
    return (
        outputs,
        _trt_last_token_logits(language_module, prefill),
        None,
        None,
    )


def run(ctx: EdgeContext) -> InferenceStageResult:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx)


def _run_eager(ctx: EdgeContext) -> InferenceStageResult:
    _prepare_language(ctx)
    eagle = ctx.model.backbone.eagle_model
    language_model = eagle.language_model
    language_inputs = ctx.inference.language_inputs
    inputs_embeds = language_inputs["inputs_embeds"]
    attention_mask = language_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = language_inputs["pad_mask"].to(dtype=torch.long)
    last_token_ids = _last_token_ids(ctx)

    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if outputs.hidden_states is None:
        raise RuntimeError("language_model returned no hidden_states")
    if outputs.logits is None:
        raise RuntimeError("language_model returned no logits")

    decoder = getattr(language_model, "model", language_model)
    hidden = outputs.hidden_states[-1]
    lm_hidden = decoder.norm(hidden) if hasattr(decoder, "norm") else hidden
    logits = gather_last_token_hidden(outputs.logits, last_token_ids).float()

    return _language_stage_result(
        ctx,
        lm_hidden_states=lm_hidden,
        logits=logits,
    )


def _run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    _prepare_language(ctx)
    module = ctx.handles.serialized.language
    if module is None:
        raise RuntimeError("serialized TRT backend missing language module")
    lm_hidden, logits, prefix_k, prefix_v = _run_trt_language(ctx, module)
    return _language_stage_result(
        ctx,
        lm_hidden_states=lm_hidden,
        logits=logits,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
    )


def _run_trt(ctx: EdgeContext) -> InferenceStageResult:
    _prepare_language(ctx)
    module = ctx.handles.in_memory.language
    if module is None:
        raise RuntimeError("in-memory TRT backend missing language module")
    lm_hidden, logits, prefix_k, prefix_v = _run_trt_language(ctx, module)
    return _language_stage_result(
        ctx,
        lm_hidden_states=lm_hidden,
        logits=logits,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
    )

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    eagle = ctx.profile.model.eagle_model
    image_token_index = getattr(
        eagle,
        "image_token_index",
        eagle.config.image_token_index,
    )

    input_ids = inputs["input_ids"]
    image_embs = inputs["image_embs"]
    input_embs = eagle.language_model.get_input_embeddings(input_ids)
    attention_mask = inputs["attention_mask"]

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    attention_mask = attention_mask.to(device=input_embs.device)

    bsz, seq_len, hidden = input_embs.shape
    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)

    image_token_mask = flat_ids == image_token_index
    flat_image_embs = image_embs.reshape(-1, hidden).to(
        device=ctx.device,
        dtype=ctx.dtype,
    )

    num_slots = int(image_token_mask.sum().item())
    if flat_image_embs.shape[0] < num_slots:
        raise ValueError(
            f"Not enough image embeddings for placeholders: "
            f"{flat_image_embs.shape[0]} embeddings for {num_slots} slots"
        )

    flat_embs[image_token_mask] = flat_image_embs[:num_slots]

    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden)
    pad_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    position_ids = torch.cumsum(pad_mask, dim=1) - 1

    return {
        "inputs_embeds": inputs_embeds,
        "pad_mask": pad_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_token_mask": image_token_mask.reshape(bsz, seq_len),
    }