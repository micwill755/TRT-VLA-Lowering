from __future__ import annotations

from trt.export.groot import build_lm_hidden_from_language_inputs
from trt.inference.backends import EagerBackend, InferenceBackend
from trt.inference.context import InferenceContext, LanguageOutputs
from trt.inference.language_prefill import build_language_prefill_inputs
from trt.language import language_head_dim
from trt.packing import pack_groot_language_inputs
from trt.runner.inference import InferenceStageResult


def run(
    ctx: InferenceContext,
    backend: InferenceBackend,
    stage_inputs: dict,
) -> InferenceStageResult:
    if not ctx.image_embs and stage_inputs.get("image_embs") is not None:
        ctx.image_embs = stage_inputs["image_embs"]

    ctx.language_inputs = pack_groot_language_inputs(
        ctx.model,
        ctx.image_embs,
        ctx.tokenized["input_ids"],
        ctx.tokenized["attention_mask"],
    )

    if isinstance(backend, EagerBackend):
        lm_hidden = build_lm_hidden_from_language_inputs(ctx.model, ctx.language_inputs)
        ctx.lm = LanguageOutputs(lm_hidden_states=lm_hidden)
    else:
        language_model = ctx.model.backbone.eagle_model.language_model
        decoder = getattr(language_model, "model", language_model)
        seq_len = int(ctx.language_inputs["inputs_embeds"].shape[1])
        max_seq_len = seq_len
        language = ctx.stage_handles.language if ctx.stage_handles else None
        if language is not None:
            max_seq_len = int(
                getattr(language, "max_seq_len", language.engine.config["max_seq_len"])
            )
        prefill = build_language_prefill_inputs(
            ctx.language_inputs,
            language_model=language_model,
            num_layers=len(decoder.layers),
            num_key_value_heads=int(language_model.config.num_key_value_heads),
            head_dim=int(language_head_dim(language_model.config)),
            max_seq_len=max(max_seq_len, seq_len),
            device=ctx.device,
        )
        ctx.lm = backend.run_language(ctx, prefill)

    lm_hidden = ctx.lm.lm_hidden_states
    if lm_hidden is None:
        raise RuntimeError("language stage did not produce lm_hidden_states")

    return InferenceStageResult(
        tensors={"lm_hidden_states": lm_hidden},
        metadata={"language_inputs": dict(ctx.language_inputs)},
    )
