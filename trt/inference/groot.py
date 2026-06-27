"""GR00T inference hooks and entrypoint wrappers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.data import pack_state
from trt.export.groot import (
    GROOT_EMBODIMENT_MAPPING,
    build_context_from_language_inputs,
    build_lm_hidden_from_language_inputs,
    make_embodiment_id,
    make_static_action_module,
    make_visual_fixed_input,
)
from trt.inference.backends import (
    EagerBackend,
    SerializedModuleBackend,
    StageHandles,
    TrtModuleBackend,
    stage_handles_from_modules,
)
from trt.inference.context import InferenceContext, LanguageOutputs
from trt.inference.hooks import VLAInferenceHooks
from trt.inference.language_prefill import build_language_prefill_inputs, run_language_prefill
from trt.inference.mode import InferenceMode
from trt.inference.pipeline import VLAInferencePipeline
from trt.io_spec import GROOT_EDGE_IO, PipelineIOSpec
from trt.language import language_head_dim
from trt.measure import compute_action_parity_metrics, tensor_error_metrics
from trt.packing import pack_groot_language_inputs
from trt.serialize import SerializedGrootLanguage
from trt.vision import nchw_to_hwc


class GrootInferenceHooks(VLAInferenceHooks):
    def preprocess(self, ctx: InferenceContext) -> None:
        tokenized_data = ctx.model_inputs["tokenized_data"]
        ctx.tokenized = {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        ctx.pixel_values = tokenized_data["pixel_values"].to(
            device=ctx.device,
            dtype=torch.float16,
        )
        state = pack_state(
            ctx.model_inputs["state"],
            max_state_dim=ctx.policy.config.max_state_dim,
            device=ctx.device,
        )
        ctx.action_side = {
            "state": state.to(device=ctx.device, dtype=torch.float16).contiguous(),
            "embodiment_id": make_embodiment_id(ctx.policy, state, ctx.device).contiguous(),
        }

    def pack_language_inputs(self, ctx: InferenceContext) -> dict:
        return pack_groot_language_inputs(
            ctx.model,
            ctx.image_embs,
            ctx.tokenized["input_ids"],
            ctx.tokenized["attention_mask"],
        )

    def language_model_for_prefill(self, ctx: InferenceContext) -> nn.Module:
        return ctx.model.backbone.eagle_model.language_model

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

    def make_rollout_noise(self, ctx: InferenceContext, context_embs: torch.Tensor) -> torch.Tensor:
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

    def action_adapter(self, ctx: InferenceContext):
        return GROOTActionAdapter(ctx.model.action_head)

    def run_eager_e2e(self, ctx: InferenceContext) -> None:
        with torch.autocast("cuda", dtype=torch.float16):
            if ctx.vision_module is None:
                visual = make_visual_fixed_input(
                    ctx.model,
                    ctx.pixel_values,
                    device=ctx.device,
                    dtype=torch.float16,
                )
                ctx.image_embs = visual(ctx.pixel_values)
            else:
                ctx.image_embs = ctx.vision_module(ctx.pixel_values)

            ctx.language_inputs = self.pack_language_inputs(ctx)
            ctx.context_embs = build_context_from_language_inputs(
                ctx.model,
                ctx.language_inputs,
            ).to(dtype=torch.float16)
            ctx.noise = self.make_rollout_noise(ctx, ctx.context_embs)

            action_module = make_static_action_module(
                ctx.model.action_head,
                ctx.device,
                torch.float16,
                ctx.action_side["embodiment_id"],
            )
            ctx.actions = sample_actions_raw(
                action_module,
                ActionRolloutContext(
                    noise=ctx.noise,
                    device=ctx.device,
                    context_embs=ctx.context_embs,
                    state=ctx.action_side["state"],
                    embodiment_id=ctx.action_side["embodiment_id"],
                ),
                self.action_adapter(ctx),
            )

    def eager_vision(self, ctx: InferenceContext, pixel_values: torch.Tensor) -> torch.Tensor:
        images_hwc = nchw_to_hwc(pixel_values.to(device=ctx.device, dtype=torch.float16).contiguous())
        visual = make_visual_fixed_input(
            ctx.model,
            images_hwc,
            device=ctx.device,
            dtype=torch.float16,
        )
        return visual(images_hwc)

    def compare_stage_parity(self, ctx: InferenceContext, *, reference_lm=None) -> None:
        del reference_lm
        print("\n=== Edge engine parity vs eager ===")
        tensor_error_metrics(
            "vision",
            ctx.image_embs.to(device=ctx.device, dtype=torch.float16),
            self.eager_vision(ctx, ctx.pixel_values).to(device=ctx.device, dtype=torch.float16),
        )
        tensor_error_metrics(
            "language lm_hidden_states",
            ctx.lm.lm_hidden_states.to(device=ctx.device, dtype=torch.float16),
            build_lm_hidden_from_language_inputs(ctx, ctx.language_inputs).to(device=ctx.device, dtype=torch.float16),
        )
        tensor_error_metrics(
            "action_context vl_embs",
            ctx.context_embs.to(device=ctx.device, dtype=torch.float16),
            build_context_from_language_inputs(ctx, ctx.language_inputs).to(device=ctx.device, dtype=torch.float16),
        )

        ctx.noise = self.make_rollout_noise(ctx, ctx.context_embs)
        timestep = torch.zeros(ctx.context_embs.shape[0], device=ctx.device, dtype=torch.long)
        eager_action = make_static_action_module(ctx)
        trt_diffusion = ctx.extras.get("trt_diffusion")
        if trt_diffusion is not None:
            with torch.no_grad():
                eager_velocity = eager_action(
                    ctx.noise,
                    timestep,
                    ctx.context_embs,
                    ctx.action_side["state"],
                    ctx.action_side["embodiment_id"],
                )
                trt_velocity = trt_diffusion(
                    ctx.noise,
                    timestep,
                    ctx.context_embs,
                    ctx.action_side["state"],
                    ctx.action_side["embodiment_id"],
                )
            tensor_error_metrics(
                "diffusion velocity",
                trt_velocity.to(device=ctx.device, dtype=torch.float16),
                eager_velocity.to(device=ctx.device, dtype=torch.float16),
            )

            eager_actions = sample_actions_raw(
                eager_action,
                ActionRolloutContext(
                    noise=ctx.noise,
                    device=ctx.device,
                    context_embs=ctx.context_embs,
                    state=ctx.action_side["state"],
                    embodiment_id=ctx.action_side["embodiment_id"],
                ),
                self.action_adapter(ctx),
            )
            trt_actions = sample_actions_raw(
                trt_diffusion,
                ActionRolloutContext(
                    noise=ctx.noise,
                    device=ctx.device,
                    context_embs=ctx.context_embs,
                    state=ctx.action_side["state"],
                    embodiment_id=ctx.action_side["embodiment_id"],
                ),
                self.action_adapter(ctx),
            )
            metrics = compute_groot_policy_action_metrics(trt_actions, eager_actions, ctx.policy)
            print(
                "full action rollout:",
                f"action_ade={metrics['action_ade']:.6f}",
                f"mean_abs={metrics['mean_abs']:.6f}",
            )


def compute_groot_policy_action_metrics(
    trt_actions: torch.Tensor,
    eager_actions: torch.Tensor,
    policy: Any,
) -> dict[str, float]:
    from lerobot.utils.constants import ACTION

    output_features = getattr(policy.config, "output_features", None)
    action_dim = None
    if output_features is not None:
        action_feature = output_features.get(ACTION)
        if action_feature is not None:
            shape = getattr(action_feature, "shape", None)
            if shape:
                action_dim = int(shape[0])
    if action_dim is not None:
        trt_actions = trt_actions[..., :action_dim]
        eager_actions = eager_actions[..., :action_dim]
    return compute_action_parity_metrics(trt_actions, eager_actions)


@torch.no_grad()
def run_serialized_language(
    engine_lm: SerializedGrootLanguage,
    model: nn.Module,
    language_inputs: dict,
    device: torch.device,
) -> torch.Tensor:
    language_model = model.backbone.eagle_model.language_model
    decoder = getattr(language_model, "model", language_model)
    seq_len = int(language_inputs["inputs_embeds"].shape[1])
    prefill = build_language_prefill_inputs(
        language_inputs,
        language_model=language_model,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(language_model.config.num_key_value_heads),
        head_dim=int(language_head_dim(language_model.config)),
        max_seq_len=seq_len,
        device=device,
    )
    outputs = run_language_prefill(engine_lm, prefill, GROOT_EDGE_IO.language)
    return outputs.lm_hidden_states


@torch.no_grad()
def run_serialized_action_context(engine_context, lm_hidden_states: torch.Tensor) -> torch.Tensor:
    return engine_context(lm_hidden_states.to(dtype=torch.float16).contiguous())


def _pipeline(io: PipelineIOSpec = GROOT_EDGE_IO) -> VLAInferencePipeline:
    return VLAInferencePipeline(GrootInferenceHooks(), io=io)


@torch.no_grad()
def run_inference_pytorch_groot(
    model,
    policy,
    model_inputs: dict,
    *,
    seed: int,
    device: torch.device,
    vision_module=None,
    io: PipelineIOSpec = GROOT_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    result = _pipeline(io).run(
        model,
        policy,
        device,
        model_inputs,
        EagerBackend(),
        seed=seed,
        vision_module=vision_module,
    )
    return result.actions, result.extras, result.elapsed_s


@torch.no_grad()
def run_inference_trt_plugin(
    model,
    policy,
    model_inputs: dict,
    *,
    trt_vision,
    trt_lm,
    trt_diffusion,
    plugin_info: dict,
    seed: int,
    device: torch.device,
    trt_action_context=None,
    io: PipelineIOSpec = GROOT_EDGE_IO,
) -> tuple[torch.Tensor, dict, float]:
    backend = TrtModuleBackend(
        stage_handles_from_modules(
            vision=trt_vision,
            language=trt_lm,
            action_context=trt_action_context,
            action=trt_diffusion,
            plugin_info=plugin_info,
        )
    )
    result = _pipeline(io).run(
        model,
        policy,
        device,
        model_inputs,
        backend,
        seed=seed,
        plugin_info=plugin_info,
    )
    return result.actions, result.extras, result.elapsed_s


@torch.no_grad()
def compare_edge_pipeline_to_eager(
    model: nn.Module,
    policy: Any,
    *,
    pixel_values: torch.Tensor,
    language_inputs: dict,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    trt_image_embs: torch.Tensor,
    lm_hidden_states: torch.Tensor,
    context_embs: torch.Tensor,
    trt_diffusion: nn.Module,
    device: torch.device,
    seed: int,
) -> None:
    del pixel_values, state, embodiment_id
    hooks = GrootInferenceHooks()
    ctx = InferenceContext(
        model=model,
        policy=policy,
        device=device,
        model_inputs={},
        io=GROOT_EDGE_IO,
        seed=seed,
    )
    ctx.image_embs = trt_image_embs
    ctx.language_inputs = language_inputs
    ctx.lm = LanguageOutputs(lm_hidden_states=lm_hidden_states)
    ctx.context_embs = context_embs
    ctx.extras["trt_diffusion"] = trt_diffusion
    hooks.compare_stage_parity(ctx)
