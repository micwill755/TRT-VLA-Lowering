"""SmolVLA inference hooks and entrypoint wrappers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trt.action_rollout import ActionRolloutContext, sample_actions_raw
from trt.export.smolvla import (
    action_output_dim,
    prepare_smolvla_batch,
    smolvla_action_adapter,
    smolvla_language_model,
    smolvla_text_config,
)
from trt.inference.backends import (
    EagerBackend,
    SerializedModuleBackend,
    TrtModuleBackend,
    stage_handles_from_modules,
)
from trt.inference.context import InferenceContext
from trt.inference.hooks import VLAInferenceHooks
from trt.inference.pipeline import VLAInferencePipeline
from trt.io_spec import PI05_EDGE_IO, PipelineIOSpec
from trt.language import language_head_dim
from trt.packing import pack_smolvla_prefix

class SmolVLAInferenceHooks(VLAInferenceHooks):
    def preprocess(self, ctx: InferenceContext) -> None:
        images, img_masks, tokens, masks, state = prepare_smolvla_batch(
            ctx.policy,
            ctx.model_inputs,
            ctx.device,
        )
        ctx.pixel_values = images[0].contiguous()
        ctx.action_side = {
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "state": state,
            "batch_size": int(tokens.shape[0]),
        }

    def pack_language_inputs(self, ctx: InferenceContext) -> dict:
        return pack_smolvla_prefix(
            ctx.model,
            ctx.image_embs,
            ctx.action_side["img_masks"],
            ctx.action_side["tokens"],
            ctx.action_side["masks"],
            ctx.action_side["state"],
        )

    def language_model_for_prefill(self, ctx: InferenceContext) -> nn.Module:
        return smolvla_language_model(ctx.model)

    def language_prefill_scalars(self, ctx: InferenceContext) -> dict[str, int]:
        cfg = smolvla_text_config(ctx.model)
        seq_len = int(ctx.language_inputs["inputs_embeds"].shape[1])
        max_seq_len = seq_len
        language = ctx.stage_handles.language if ctx.stage_handles else None
        if language is not None:
            max_seq_len = int(getattr(language, "max_seq_len", language.engine.config["max_seq_len"]))
        return {
            "num_layers": int(ctx.model.vlm_with_expert.num_vlm_layers),
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "head_dim": int(language_head_dim(cfg)),
            "max_seq_len": max(max_seq_len, seq_len),
        }

    def make_rollout_noise(self, ctx: InferenceContext, _reference: torch.Tensor) -> torch.Tensor:
        core = ctx.model
        state = ctx.action_side["state"]
        return core.sample_noise(
            (state.shape[0], core.config.chunk_size, core.config.max_action_dim),
            ctx.device,
        )

    def action_adapter(self, ctx: InferenceContext):
        return smolvla_action_adapter(ctx.model)

    def build_action_rollout_context(
        self,
        ctx: InferenceContext,
        noise: torch.Tensor,
        *,
        context_embs: torch.Tensor | None = None,
    ):
        del context_embs
        return ActionRolloutContext(
            noise=noise,
            device=ctx.device,
            prefix_k=ctx.action_side["prefix_k"],
            prefix_v=ctx.action_side["prefix_v"],
            prefix_pad_mask=ctx.action_side["prefix_pad_mask"],
        )

    def run_vision_embs(self, ctx: InferenceContext, backend) -> list[torch.Tensor]:
        return [backend.run_vision(ctx, image.to(device=ctx.device)) for image in ctx.action_side["images"]]

    def run_eager_e2e(self, ctx: InferenceContext) -> None:
        images = ctx.action_side["images"]
        img_masks = ctx.action_side["img_masks"]
        tokens = ctx.action_side["tokens"]
        masks = ctx.action_side["masks"]
        state = ctx.action_side["state"]
        ctx.noise = self.make_rollout_noise(ctx, state)
        ctx.actions = ctx.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            state,
            noise=ctx.noise,
        )
        
        ctx.actions = ctx.actions[..., : action_output_dim(ctx.policy)]

@torch.no_grad()
def run_inference_pytorch_smolvla(
    model,
    policy,
    batch: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    vision_module=None,
    io: PipelineIOSpec = PI05_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    del vision_module
    result = VLAInferencePipeline(SmolVLAInferenceHooks(), io=io).run(
        model,
        policy,
        device,
        batch,
        EagerBackend(),
        seed=seed,
    )
    return result.actions, result.extras, result.elapsed_s


@torch.no_grad()
def run_inference_smolvla_engines(
    model,
    policy,
    batch: dict[str, Any],
    *,
    vision_runner,
    language_runner,
    diffusion_runner,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    backend = SerializedModuleBackend(
        stage_handles_from_modules(
            vision=vision_runner,
            language=language_runner,
            action=diffusion_runner,
        )
    )
    result = VLAInferencePipeline(SmolVLAInferenceHooks(), io=io).run(
        model,
        policy,
        device,
        batch,
        backend,
        seed=seed,
    )
    result.actions = result.actions[..., : action_output_dim(policy)]
    return result.actions, result.extras, result.elapsed_s
