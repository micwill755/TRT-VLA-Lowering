from __future__ import annotations

from trt.export.groot import build_context_from_language_inputs
from trt.inference.backends import EagerBackend, InferenceBackend
from trt.inference.context import InferenceContext
from trt.runner.inference import InferenceStageResult


def run(
    ctx: InferenceContext,
    backend: InferenceBackend,
    stage_inputs: dict,
) -> InferenceStageResult:
    lm_hidden = stage_inputs.get("lm_hidden_states")
    if lm_hidden is None and ctx.lm is not None:
        lm_hidden = ctx.lm.lm_hidden_states
    if lm_hidden is None:
        raise RuntimeError("action_context stage requires language lm_hidden_states")

    if isinstance(backend, EagerBackend):
        context_embs = build_context_from_language_inputs(ctx.model, ctx.language_inputs)
    elif backend.has_action_context():
        context_embs = backend.run_action_context(ctx, lm_hidden)
    else:
        context_embs = lm_hidden

    ctx.context_embs = context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    return InferenceStageResult(tensors={"context_embs": ctx.context_embs})
