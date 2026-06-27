"""MolmoAct2-specific export hooks, TRT modules, and helpers."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from lerobot.utils.constants import ACTION

from trt.action_rollout import ActionRolloutContext, EncoderKVFlowActionAdapter, sample_actions_raw
from trt.diffusion import DiffusionEngineSpec
from trt.export.context import ComponentBuild, ExportContext
from trt.export.hooks import VLAExportHooks
from trt.export.mode import ExportMode
from trt.export.settings import ACTION_TRT_SETTINGS
from trt.export.sinks import ExportSink
from trt.io_spec import (
    MOLMOACT2_ACTION_ROLLOUT,
    MOLMOACT2_EDGE_IO,
    PipelineIOSpec,
    action_rollout_extra_config,
)
from trt.measure import compute_action_parity_metrics, tensor_error_metrics
from trt.serialize import SerializedTRTEngine

MODEL_INPUT_KEYS = (
    "input_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "attention_mask",
    "position_ids",
    "token_type_ids",
)


def molmoact2_model_inputs(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in MODEL_INPUT_KEYS:
        value = batch.get(key)
        if value is None:
            continue
        if value.is_floating_point():
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def stack_encoder_kv(
    kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_k = torch.stack([item[0] for item in kv_states], dim=0).contiguous()
    encoder_v = torch.stack([item[1] for item in kv_states], dim=0).contiguous()
    return encoder_k, encoder_v


def unstack_encoder_kv(
    encoder_k: torch.Tensor,
    encoder_v: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(encoder_k[i].contiguous(), encoder_v[i].contiguous()) for i in range(int(encoder_k.shape[0]))]


def action_output_dim(policy: Any) -> int:
    output_feature = policy.config.output_features.get(ACTION)
    if output_feature is not None and output_feature.shape:
        return int(output_feature.shape[0])
    return int(getattr(policy.config, "expected_max_action_dim", 32))


def crop_policy_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    return actions[..., : action_output_dim(policy)]


def flow_matching_steps(policy: Any) -> int:
    backbone = policy._backbone()
    return int(getattr(backbone.config, "flow_matching_num_steps", policy.config.num_inference_steps or 10))


class MolmoAct2BackboneKVModule(nn.Module):
    """Fused MolmoAct2 multimodal prefill returning stacked encoder K/V."""

    def __init__(self, policy: Any):
        super().__init__()
        self.policy = policy
        self.backbone = policy._backbone()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        image_num_crops: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_kv = self.backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self.policy._encoder_attention_mask_for_action_expert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        depth_gate, depth_mask = self.backbone._depth_gate_from_condition(
            input_ids=input_ids,
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv,
        )
        encoder_kv = self.backbone._apply_depth_gate_to_layer_kv_states(
            encoder_kv,
            depth_mask,
            depth_gate,
        )
        return stack_encoder_kv(encoder_kv)


class MolmoAct2ActionFlowStepModule(nn.Module):
    """Single continuous flow-matching velocity step with baked action-expert context."""

    def __init__(self, policy: Any):
        super().__init__()
        self.action_expert = policy._action_expert()

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(x_t.shape[0])
        seq_len = int(x_t.shape[1])
        device = x_t.device
        dtype = x_t.dtype
        encoder_kv = unstack_encoder_kv(encoder_k, encoder_v)
        context = self.action_expert.prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=encoder_attention_mask,
            action_attention_mask=None,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        modulation = self.action_expert.prepare_modulation_cache([timestep.to(dtype=dtype)])[0]
        return self.action_expert.forward_with_context(
            x_t,
            timestep.to(dtype=dtype),
            context=context,
            modulation=modulation,
        )


class SerializedMolmoAct2Backbone:
    def __init__(self, engine: SerializedTRTEngine):
        self.engine = engine

    def __call__(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        image_num_crops: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.engine(
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "image_token_pooling": image_token_pooling,
                "image_grids": image_grids,
                "image_num_crops": image_num_crops,
                "attention_mask": attention_mask,
            }
        )
        return outputs[0], outputs[1]


class SerializedMolmoAct2Action:
    def __init__(self, engine: SerializedTRTEngine):
        self.engine = engine

    def __call__(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.engine(
            {
                "x_t": x_t,
                "timestep": timestep,
                "encoder_k": encoder_k,
                "encoder_v": encoder_v,
                "encoder_attention_mask": encoder_attention_mask,
            }
        )[0]


@torch.no_grad()
def compare_molmoact2_edge_pipeline_to_eager(
    policy: Any,
    *,
    model_inputs: dict[str, torch.Tensor],
    trt_encoder_k: torch.Tensor,
    trt_encoder_v: torch.Tensor,
    action_runner: nn.Module,
    device: torch.device,
    seed: int,
) -> None:
    print("\n=== MolmoAct2 Edge engine parity vs eager")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dtype = trt_encoder_k.dtype
    eager_inputs = molmoact2_model_inputs(model_inputs, device=device, dtype=dtype)
    eager_module = MolmoAct2BackboneKVModule(policy).eval().to(device)
    eager_k, eager_v = eager_module(
        eager_inputs["input_ids"],
        eager_inputs["pixel_values"],
        eager_inputs["image_token_pooling"],
        eager_inputs["image_grids"],
        eager_inputs["image_num_crops"],
        eager_inputs["attention_mask"],
    )
    tensor_error_metrics("backbone encoder_k", trt_encoder_k, eager_k)
    tensor_error_metrics("backbone encoder_v", trt_encoder_v, eager_v)

    batch = dict(model_inputs)
    batch.setdefault("action_dim_is_pad", model_inputs.get("action_dim_is_pad"))
    eager_actions = policy.predict_action_chunk(
        batch,
        generator=torch.Generator(device=device).manual_seed(seed),
        inference_action_mode="continuous",
    )
    encoder_attention_mask = policy._encoder_attention_mask_for_action_expert(
        input_ids=eager_inputs["input_ids"],
        attention_mask=eager_inputs.get("attention_mask"),
    )
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(
            eager_inputs["input_ids"].shape,
            device=device,
            dtype=torch.bool,
        )
    else:
        encoder_attention_mask = encoder_attention_mask.to(device=device)

    noise = torch.randn(
        eager_actions.shape[0],
        policy._generation_action_horizon(),
        policy._backbone().config.max_action_dim,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed + 1),
    )
    rollout_ctx = ActionRolloutContext(
        noise=noise,
        device=device,
        prefix_k=trt_encoder_k,
        prefix_v=trt_encoder_v,
        encoder_attention_mask=encoder_attention_mask.to(dtype=dtype),
    )
    trt_actions = sample_actions_raw(
        action_runner,
        rollout_ctx,
        EncoderKVFlowActionAdapter(
            num_steps_value=flow_matching_steps(policy),
            dt_sign=1,
        ),
    )

    metrics = compute_action_parity_metrics(
        crop_policy_actions(policy, trt_actions),
        crop_policy_actions(policy, eager_actions),
    )
    print(
        f"  action parity: ADE={metrics['action_ade']:.6f}  "
        f"mean_abs={metrics['mean_abs']:.6f}"
    )


class MolmoAct2ExportHooks(VLAExportHooks):
    def __init__(
        self,
        *,
        io: PipelineIOSpec = MOLMOACT2_EDGE_IO,
        action_trt_settings: dict | None = None,
    ) -> None:
        self.io = io
        self.tokenizer = None
        self.action_trt_settings = action_trt_settings or dict(ACTION_TRT_SETTINGS)

    def preprocess(self, ctx: ExportContext) -> None:
        policy = ctx.policy
        if policy.config.inference_action_mode is None:
            policy.config.inference_action_mode = "continuous"
        dtype = torch.bfloat16 if policy.config.model_dtype == "bfloat16" else torch.float16
        ctx.action_side = {
            "export_dtype": dtype,
            "batch_size": int(next(iter(ctx.model_inputs.values())).shape[0]),
        }

    def build_vision_spec(self, ctx: ExportContext):
        raise RuntimeError("MolmoAct2 does not export a standalone vision engine")

    def pack_language_inputs(self, ctx: ExportContext) -> dict:
        dtype = ctx.action_side["export_dtype"]
        return molmoact2_model_inputs(ctx.model_inputs, device=ctx.device, dtype=dtype)

    def build_language_spec(self, ctx: ExportContext):
        raise RuntimeError("MolmoAct2 uses build_backbone_component instead of LanguageEngineSpec")

    def build_chat_template(self, tokenizer: Any) -> dict[str, Any]:
        del tokenizer
        return {}

    def save_language_artifacts(self, ctx: ExportContext, language_dir) -> None:
        del ctx, language_dir

    def build_backbone_component(self, ctx: ExportContext) -> ComponentBuild:
        policy = ctx.policy
        language_inputs = self.pack_language_inputs(ctx)
        required = (
            "input_ids",
            "pixel_values",
            "image_token_pooling",
            "image_grids",
            "image_num_crops",
        )
        missing = [key for key in required if key not in language_inputs]
        if missing:
            raise KeyError(f"MolmoAct2 compile batch missing required keys: {missing}")

        attention_mask = language_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(language_inputs["input_ids"], dtype=torch.bool)

        sample_inputs = (
            language_inputs["input_ids"],
            language_inputs["pixel_values"],
            language_inputs["image_token_pooling"],
            language_inputs["image_grids"],
            language_inputs["image_num_crops"],
            attention_mask,
        )
        module = MolmoAct2BackboneKVModule(policy).eval().to(ctx.device)
        with torch.no_grad():
            encoder_k, encoder_v = module(*sample_inputs)
        encoder_attention_mask = policy._encoder_attention_mask_for_action_expert(
            input_ids=language_inputs["input_ids"],
            attention_mask=attention_mask,
        )
        if encoder_attention_mask is None:
            encoder_attention_mask = attention_mask.to(dtype=torch.bool)
        ctx.action_side["encoder_attention_mask_tensor"] = encoder_attention_mask.contiguous()

        num_layers = int(encoder_k.shape[0])
        seq_len = int(encoder_k.shape[-2])
        return ComponentBuild(
            module=module,
            sample_inputs=sample_inputs,
            extra_config={
                "engine_role": "molmoact2_backbone_kv",
                "num_layers": num_layers,
                "encoder_seq_len": seq_len,
            },
            trt_settings=self.action_trt_settings,
            model_type="molmoact2_backbone",
            component="language",
            engine_file="backbone.engine",
        )

    def build_diffusion_spec(self, ctx: ExportContext):
        policy = ctx.policy
        encoder_k = ctx.action_side["encoder_k"]
        encoder_v = ctx.action_side["encoder_v"]
        encoder_attention_mask = ctx.action_side["encoder_attention_mask_tensor"]
        action_module = MolmoAct2ActionFlowStepModule(policy).eval().to(ctx.device)
        batch_size = int(ctx.action_side["batch_size"])
        action_horizon = int(policy._generation_action_horizon())
        max_action_dim = int(policy._backbone().config.max_action_dim)
        dtype = ctx.action_side["export_dtype"]
        x_t = torch.randn(batch_size, action_horizon, max_action_dim, device=ctx.device, dtype=dtype)
        timestep = torch.zeros(batch_size, device=ctx.device, dtype=dtype)
        sample_inputs = (
            x_t,
            timestep,
            encoder_k.to(device=ctx.device, dtype=dtype),
            encoder_v.to(device=ctx.device, dtype=dtype),
            encoder_attention_mask.to(device=ctx.device, dtype=dtype),
        )
        return DiffusionEngineSpec(
            diffusion_module=action_module,
            sample_inputs=sample_inputs,
            extra_config={
                "engine_role": "single_action_denoising_step",
                **action_rollout_extra_config(
                    self.io,
                    MOLMOACT2_ACTION_ROLLOUT,
                    num_steps=flow_matching_steps(policy),
                    chunk_size=action_horizon,
                    max_action_dim=max_action_dim,
                    prefix_seq_len=int(encoder_k.shape[-2]),
                    num_layers=int(encoder_k.shape[0]),
                ),
            },
            io=self.io.action,
            trt_settings=self.action_trt_settings,
            model_type="molmoact2_action",
            engine_file="diffusion.engine",
        )

    def after_export(self, ctx: ExportContext, sink: ExportSink) -> None:
        if not ctx.accuracy_check or sink.mode is not ExportMode.SERIALIZED:
            return
        action_runner = ctx.handles.get("action")
        if action_runner is None:
            return
        compare_molmoact2_edge_pipeline_to_eager(
            ctx.policy,
            model_inputs=ctx.model_inputs,
            trt_encoder_k=ctx.action_side["encoder_k"],
            trt_encoder_v=ctx.action_side["encoder_v"],
            action_runner=action_runner,
            device=ctx.device,
            seed=ctx.seed,
        )
