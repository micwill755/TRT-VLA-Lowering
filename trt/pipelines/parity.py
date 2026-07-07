from __future__ import annotations

from trt.config.parity_mode import ParityMode
from trt.context import EdgeContext

# current stage -> (reference stage, reference tensor key, key in inputs["tensors"])
UPSTREAM_TENSOR_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "language": ("vision", "image_embs", "image_embs"),
    "action_context": ("language", "lm_hidden", "lm_hidden"),
    "action": ("action_context", "context_embs", "context_embs"),
}

# Top-level pipeline keys pinned from parity_reference["action"] in isolated mode.
ACTION_SIDE_KEYS = ("state", "embodiment_id")


def _is_isolated(ctx: EdgeContext) -> bool:
    return ctx.parity_active == ParityMode.ISOLATED.value


def maybe_override_upstream(ctx: EdgeContext, stage_name: str, inputs: dict) -> dict:
    """Replace upstream stage tensors with eager reference values when isolated."""
    if not _is_isolated(ctx):
        return inputs

    spec = UPSTREAM_TENSOR_OVERRIDES.get(stage_name)
    if spec is None:
        return inputs

    ref_stage, ref_key, input_key = spec
    ref = ctx.parity_reference.get(ref_stage, {}).get(ref_key)
    if ref is None:
        return inputs

    out = dict(inputs)
    tensors = dict(out.get("tensors", {}))
    tensors[input_key] = ref
    out["tensors"] = tensors
    return out


def maybe_override_action_side(ctx: EdgeContext, inputs: dict) -> dict:
    """Pin state and embodiment_id for isolated action parity."""
    if not _is_isolated(ctx):
        return inputs

    ref = ctx.parity_reference.get("action", {})
    if not ref:
        return inputs

    out = dict(inputs)
    for key in ACTION_SIDE_KEYS:
        value = ref.get(key)
        if value is not None:
            out[key] = value
    return out


def parity_initial_actions(ctx: EdgeContext):
    """Return eager initial diffusion noise when running isolated action parity."""
    if not _is_isolated(ctx):
        return None
    return ctx.parity_reference.get("action", {}).get("initial_actions")
