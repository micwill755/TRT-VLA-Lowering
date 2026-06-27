"""MolmoAct2 inference hooks and entrypoint wrappers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trt.action_rollout import ActionRolloutContext, EncoderKVFlowActionAdapter
from trt.export.molmoact2 import (
    action_output_dim,
    crop_policy_actions,
    flow_matching_steps,
    molmoact2_model_inputs,
)
from trt.inference.backends import (
    EagerBackend,
    SerializedModuleBackend,
    TrtModuleBackend,
    stage_handles_from_modules,
)
from trt.inference.context import InferenceContext
from trt.inference.hooks import VLAInferenceHooks
from trt.inference.molmoact2_pipeline import MolmoAct2InferencePipeline
from trt.io_spec import MOLMOACT2_EDGE_IO, PipelineIOSpec


class MolmoAct2InferenceHooks(VLAInferenceHooks):
    def preprocess(self, ctx: InferenceContext) -> None:
        policy = ctx.policy
        if policy.config.inference_action_mode is None:
            policy.config.inference_action_mode = "continuous"
        dtype = (
            torch.bfloat16
            if policy.config.model_dtype == "bfloat16"
            else torch.float16
        )
        batch_size = int(next(iter(ctx.model_inputs.values())).shape[0])
        ctx.action_side = {
            "export_dtype": dtype,
            "batch_size": batch_size,
        }

    def pack_language_inputs(self, ctx: InferenceContext) -> dict:
        return molmoact2_model_inputs(
            ctx.model_inputs,
            device=ctx.device,
            dtype=ctx.action_side["export_dtype"],
        )

    def language_model_for_prefill(self, ctx: InferenceContext) -> nn.Module:
        return ctx.policy._backbone()

    def language_prefill_scalars(self, ctx: InferenceContext) -> dict[str, int]:
        del ctx
        return {
            "num_layers": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "max_seq_len": 1,
        }

    def make_rollout_noise(self, ctx: InferenceContext, _reference: torch.Tensor) -> torch.Tensor:
        policy = ctx.policy
        batch_size = int(ctx.action_side["batch_size"])
        action_horizon = int(policy._generation_action_horizon())
        max_action_dim = int(policy._backbone().config.max_action_dim)
        dtype = ctx.action_side["export_dtype"]
        return torch.randn(
            batch_size,
            action_horizon,
            max_action_dim,
            device=ctx.device,
            dtype=dtype,
        )

    def action_adapter(self, ctx: InferenceContext):
        num_steps = flow_matching_steps(ctx.policy)
        action = ctx.stage_handles.action if ctx.stage_handles else None
        if action is not None and getattr(action, "engine", None) is not None:
            num_steps = int(action.engine.config.get("num_inference_steps", num_steps))
        return EncoderKVFlowActionAdapter(num_steps_value=num_steps, dt_sign=1)

    def build_action_rollout_context(
        self,
        ctx: InferenceContext,
        noise: torch.Tensor,
        *,
        context_embs: torch.Tensor | None = None,
    ):
        del context_embs
        encoder_attention_mask = ctx.action_side.get("encoder_attention_mask")
        if encoder_attention_mask is None:
            raise RuntimeError("MolmoAct2 action rollout requires encoder_attention_mask")
        return ActionRolloutContext(
            noise=noise,
            device=ctx.device,
            prefix_k=ctx.action_side["encoder_k"],
            prefix_v=ctx.action_side["encoder_v"],
            encoder_attention_mask=encoder_attention_mask,
        )

    def run_eager_e2e(self, ctx: InferenceContext) -> None:
        batch = dict(ctx.model_inputs)
        ctx.actions = ctx.policy.predict_action_chunk(
            batch,
            inference_action_mode="continuous",
            generator=torch.Generator(device=ctx.device).manual_seed(ctx.seed),
        )
        ctx.actions = crop_policy_actions(ctx.policy, ctx.actions)

    def finalize_extras(self, ctx: InferenceContext) -> dict[str, Any]:
        extras = dict(ctx.extras)
        if ctx.actions is not None:
            extras["output_action_dim"] = action_output_dim(ctx.policy)
        return extras


def _pipeline(io: PipelineIOSpec = MOLMOACT2_EDGE_IO) -> MolmoAct2InferencePipeline:
    return MolmoAct2InferencePipeline(MolmoAct2InferenceHooks(), io=io)


@torch.no_grad()
def run_inference_pytorch_molmoact2(
    model,
    policy,
    batch: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = MOLMOACT2_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    del model
    result = _pipeline(io).run(
        model,
        policy,
        device,
        batch,
        EagerBackend(),
        seed=seed,
    )
    return result.actions, result.extras, result.elapsed_s


@torch.no_grad()
def run_inference_molmoact2_engines(
    model,
    policy,
    batch: dict[str, Any],
    *,
    backbone_runner,
    diffusion_runner,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = MOLMOACT2_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    backend = SerializedModuleBackend(
        stage_handles_from_modules(
            language=backbone_runner,
            action=diffusion_runner,
        )
    )
    result = _pipeline(io).run(
        model,
        policy,
        device,
        batch,
        backend,
        seed=seed,
    )
    result.actions = crop_policy_actions(policy, result.actions)
    return result.actions, result.extras, result.elapsed_s


@torch.no_grad()
def run_inference_trt_molmoact2(
    model,
    policy,
    batch: dict[str, Any],
    *,
    trt_backbone,
    trt_diffusion,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = MOLMOACT2_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    backend = TrtModuleBackend(
        stage_handles_from_modules(
            language=trt_backbone,
            action=trt_diffusion,
        )
    )
    result = _pipeline(io).run(
        model,
        policy,
        device,
        batch,
        backend,
        seed=seed,
    )
    result.actions = crop_policy_actions(policy, result.actions)
    return result.actions, result.extras, result.elapsed_s
