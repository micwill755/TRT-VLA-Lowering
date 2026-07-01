from __future__ import annotations

import torch

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.export.groot import make_static_action_module
from trt.inference.backends import EagerBackend, InferenceBackend
from trt.inference.context import InferenceContext
from trt.runner.inference import InferenceStageResult


class _ActionRolloutHooks:
    @staticmethod
    def build_action_rollout_context(ctx: InferenceContext, noise: torch.Tensor, *, context_embs):
        return ActionRolloutContext(
            noise=noise,
            device=ctx.device,
            context_embs=context_embs,
            state=ctx.action_side.get("state"),
            embodiment_id=ctx.action_side.get("embodiment_id"),
        )

    @staticmethod
    def action_adapter(ctx: InferenceContext):
        return GROOTActionAdapter(ctx.model.action_head)


def _make_rollout_noise(ctx: InferenceContext, context_embs: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(ctx.seed)
    return torch.randn(
        context_embs.shape[0],
        ctx.model.action_head.config.action_horizon,
        ctx.model.action_head.config.action_dim,
        device=ctx.device,
        dtype=context_embs.dtype,
        generator=generator,
    )


def run(
    ctx: InferenceContext,
    backend: InferenceBackend,
    stage_inputs: dict,
) -> InferenceStageResult:
    del stage_inputs
    if ctx.context_embs is None:
        raise RuntimeError("action stage requires context_embs")

    ctx.context_embs = ctx.context_embs.to(device=ctx.device, dtype=torch.float16).contiguous()
    ctx.noise = _make_rollout_noise(ctx, ctx.context_embs)
    rollout_hooks = _ActionRolloutHooks()

    if isinstance(backend, EagerBackend):
        with torch.autocast("cuda", dtype=torch.float16):
            action_module = make_static_action_module(
                ctx.model.action_head,
                ctx.device,
                torch.float16,
                ctx.action_side["embodiment_id"],
            )
            ctx.actions = sample_actions_raw(
                action_module,
                rollout_hooks.build_action_rollout_context(
                    ctx,
                    ctx.noise,
                    context_embs=ctx.context_embs,
                ),
                rollout_hooks.action_adapter(ctx),
            )
    else:
        ctx.actions = backend.run_action_rollout(
            ctx,
            ctx.context_embs,
            ctx.noise,
            rollout_hooks,
        )

    return InferenceStageResult(tensors={"actions": ctx.actions})
