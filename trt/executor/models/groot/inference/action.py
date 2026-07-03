from __future__ import annotations

import torch

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.modules.export.diffusion import (
    GrootDiTStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
    TRTDynamicCategorySpecificMLPExportModule,
)
from trt.runner.inference import InferenceStageResult


class _ActionRolloutHooks:
    @staticmethod
    def build_action_rollout_context(ctx: EdgeContext, noise: torch.Tensor, *, context_embs):
        return ActionRolloutContext(
            noise=noise,
            device=ctx.device,
            context_embs=context_embs,
            state=ctx.inference.action_side.get("state"),
            embodiment_id=ctx.inference.action_side.get("embodiment_id"),
        )

    @staticmethod
    def action_adapter(ctx: EdgeContext):
        return GROOTActionAdapter(ctx.model.action_head)


def _make_rollout_noise(ctx: EdgeContext, context_embs: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(ctx.inference.seed)
    return torch.randn(
        context_embs.shape[0],
        ctx.model.action_head.config.action_horizon,
        ctx.model.action_head.config.action_dim,
        device=ctx.device,
        dtype=context_embs.dtype,
        generator=generator,
    )


def _prepare_action(ctx: EdgeContext) -> _ActionRolloutHooks:
    if ctx.inference.context_embs is None:
        raise RuntimeError("action stage requires context_embs")
    ctx.inference.context_embs = ctx.inference.context_embs.to(
        device=ctx.device,
        dtype=torch.float16,
    ).contiguous()
    ctx.inference.noise = _make_rollout_noise(ctx, ctx.inference.context_embs)
    return _ActionRolloutHooks()


def run(ctx: EdgeContext) -> InferenceStageResult:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx)


def _run_eager(ctx: EdgeContext) -> InferenceStageResult:
    rollout_hooks = _prepare_action(ctx)
    with torch.autocast("cuda", dtype=torch.float16):
        action_head = ctx.model.action_head
        embodiment_id = ctx.inference.action_side["embodiment_id"]
        velocity_decoder = action_head.action_decoder
        if embodiment_id is not None:
            velocity_decoder = TRTDynamicCategorySpecificMLPExportModule(
                action_head.action_decoder
            )
        action_module = StaticActionVelocityStepExportModule(
            step_encoder=GrootDiTStepEncoderExportModule(action_head, embodiment_id),
            action_expert=action_head.model,
            velocity_decoder=velocity_decoder,
            output_tokens=action_head.config.action_horizon,
            cast_hidden_fp32=False,
        ).eval().to(device=ctx.device, dtype=torch.float16)
        actions = sample_actions_raw(
            action_module,
            rollout_hooks.build_action_rollout_context(
                ctx,
                ctx.inference.noise,
                context_embs=ctx.inference.context_embs,
            ),
            rollout_hooks.action_adapter(ctx),
        )
    ctx.actions = actions
    return InferenceStageResult(tensors={"actions": actions})


def _run_serialized(ctx: EdgeContext) -> InferenceStageResult:
    rollout_hooks = _prepare_action(ctx)
    action_module = ctx.handles.serialized.action
    if action_module is None:
        raise RuntimeError("serialized TRT backend missing action module")
    actions = sample_actions_raw(
        action_module,
        rollout_hooks.build_action_rollout_context(
            ctx,
            ctx.inference.noise,
            context_embs=ctx.inference.context_embs,
        ),
        rollout_hooks.action_adapter(ctx),
    )
    ctx.actions = actions
    return InferenceStageResult(tensors={"actions": actions})


def _run_trt(ctx: EdgeContext) -> InferenceStageResult:
    rollout_hooks = _prepare_action(ctx)
    action_module = ctx.handles.in_memory.action
    if action_module is None:
        raise RuntimeError("in-memory TRT backend missing action module")
    actions = sample_actions_raw(
        action_module,
        rollout_hooks.build_action_rollout_context(
            ctx,
            ctx.inference.noise,
            context_embs=ctx.inference.context_embs,
        ),
        rollout_hooks.action_adapter(ctx),
    )
    ctx.actions = actions
    return InferenceStageResult(tensors={"actions": actions})
