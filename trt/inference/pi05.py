"""PI0.5 inference hooks and entrypoint wrappers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trt.action_rollout import ActionRolloutContext, PrefixKVFlowActionAdapter, sample_actions_raw
from trt.diffusion_builders import make_pi05_static_action_module
from trt.export.pi05 import crop_policy_actions
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
from trt.language import language_head_dim, run_prefix_language_eager
from trt.packing import pack_pi05_prefix
from trt.utils import ensure_pi05_paligemma_on_device, prepare_policy_inputs
from trt.vision import run_trt_vision_nchw


class Pi05InferenceHooks(VLAInferenceHooks):
    def preprocess(self, ctx: InferenceContext) -> None:
        images, img_masks, tokens, masks = prepare_policy_inputs(
            ctx.policy,
            ctx.model_inputs,
            ctx.device,
        )
        ctx.pixel_values = images[0].to(device=ctx.device).contiguous()
        ctx.action_side = {
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "batch_size": int(tokens.shape[0]),
        }
        ensure_pi05_paligemma_on_device(ctx.model, ctx.device)

    def pack_language_inputs(self, ctx: InferenceContext) -> dict:
        return pack_pi05_prefix(
            ctx.model,
            ctx.image_embs,
            ctx.action_side["img_masks"],
            ctx.action_side["tokens"],
            ctx.action_side["masks"],
            inputs_dtype=torch.float16,
        )

    def language_model_for_prefill(self, ctx: InferenceContext) -> nn.Module:
        return ctx.model.paligemma_with_expert.paligemma.model.language_model

    def language_prefill_scalars(self, ctx: InferenceContext) -> dict[str, int]:
        lm = self.language_model_for_prefill(ctx)
        decoder = getattr(lm, "model", lm)
        seq_len = int(ctx.language_inputs["inputs_embeds"].shape[1])
        max_seq_len = int(ctx.plugin_info.get("language_max_seq_len", seq_len))
        return {
            "num_layers": len(decoder.layers),
            "num_key_value_heads": int(lm.config.num_key_value_heads),
            "head_dim": int(language_head_dim(lm.config)),
            "max_seq_len": max(max_seq_len, seq_len),
        }

    def make_rollout_noise(self, ctx: InferenceContext, _reference: torch.Tensor) -> torch.Tensor:
        core = ctx.model
        batch_size = int(ctx.action_side["batch_size"])
        return core.sample_noise(
            (batch_size, core.config.chunk_size, core.config.max_action_dim),
            ctx.device,
        )

    def action_adapter(self, ctx: InferenceContext):
        num_steps = int(ctx.plugin_info.get("num_inference_steps", ctx.model.config.num_inference_steps))
        return PrefixKVFlowActionAdapter(ctx.model, num_steps)

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
        images = ctx.action_side["images"]
        if ctx.vision_module is not None:
            return [run_trt_vision_nchw(ctx.vision_module, image.to(device=ctx.device)) for image in images]
        return [backend.run_vision(ctx, image.to(device=ctx.device)) for image in images]

    def run_eager_e2e(self, ctx: InferenceContext) -> None:
        images = ctx.action_side["images"]
        image_embs = [ctx.model.paligemma_with_expert.embed_image(image) for image in images]
        ctx.image_embs = image_embs
        ctx.language_inputs = self.pack_language_inputs(ctx)
        _, prefix_k, prefix_v = run_prefix_language_eager(
            self.language_model_for_prefill(ctx),
            ctx.language_inputs["inputs_embeds"],
            ctx.language_inputs["attention_mask"],
            ctx.language_inputs["position_ids"],
        )
        ctx.action_side["prefix_k"] = prefix_k
        ctx.action_side["prefix_v"] = prefix_v
        ctx.action_side["prefix_pad_mask"] = ctx.language_inputs["pad_mask"]
        ctx.noise = self.make_rollout_noise(ctx, prefix_k)
        action_module = make_pi05_static_action_module(ctx.model, ctx.device)
        ctx.actions = sample_actions_raw(
            action_module,
            self.build_action_rollout_context(ctx, ctx.noise),
            self.action_adapter(ctx),
        )
        ctx.actions = crop_policy_actions(ctx.policy, ctx.actions)

    def finalize_extras(self, ctx: InferenceContext) -> dict[str, Any]:
        extras = dict(ctx.extras)
        extras.update(
            {
                "prefix_k": ctx.action_side.get("prefix_k"),
                "prefix_v": ctx.action_side.get("prefix_v"),
                "prefix_pad_mask": ctx.action_side.get("prefix_pad_mask"),
            }
        )
        return extras


def _pipeline(io: PipelineIOSpec = PI05_EDGE_IO) -> VLAInferencePipeline:
    return VLAInferencePipeline(Pi05InferenceHooks(), io=io)


@torch.no_grad()
def run_inference_pytorch_pi05(
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
def run_inference_trt_plugin(
    model,
    policy,
    batch: dict[str, Any],
    *,
    trt_vision,
    trt_lm,
    trt_diffusion,
    plugin_info: dict,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    backend = TrtModuleBackend(
        stage_handles_from_modules(
            vision=trt_vision,
            language=trt_lm,
            action=trt_diffusion,
            plugin_info=plugin_info,
        )
    )
    result = _pipeline(io).run(
        model,
        policy,
        device,
        batch,
        backend,
        seed=seed,
        plugin_info=plugin_info,
    )
    result.actions = crop_policy_actions(policy, result.actions)
    return result.actions, result.extras, result.elapsed_s


@torch.no_grad()
def run_inference_pi05_engines(
    model,
    policy,
    batch: dict[str, Any],
    *,
    vision_runner,
    language_runner,
    diffusion_runner,
    plugin_info: dict,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    backend = SerializedModuleBackend(
        stage_handles_from_modules(
            vision=vision_runner,
            language=language_runner,
            action=diffusion_runner,
            plugin_info=plugin_info,
        )
    )
    result = _pipeline(io).run(
        model,
        policy,
        device,
        batch,
        backend,
        seed=seed,
        plugin_info=plugin_info,
    )
    result.actions = crop_policy_actions(policy, result.actions)
    return result.actions, result.extras, result.elapsed_s
