"""Per-model builders for ``DiffusionEngineSpec``."""

from __future__ import annotations

import torch
import torch.nn as nn

from trt.diffusion import (
    DEFAULT_DIFFUSION_TRT_SETTINGS,
    DiffusionEngineSpec,
    GrootDiTStepEncoder,
    PI05PrefixKVStepEncoder,
    StaticActionVelocityStep,
    TRTDynamicCategorySpecificMLP,
)
from trt.io_spec import (
    GROOT_ACTION_ROLLOUT,
    GROOT_EDGE_IO,
    PI05_ACTION_ROLLOUT,
    PI05_EDGE_IO,
    PipelineIOSpec,
    action_rollout_extra_config,
)
from trt.utils import make_suffix_position_and_mask


def make_groot_action_compile_inputs(
    action_horizon: int,
    action_dim: int,
    vl_embs: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    batch_size = vl_embs.shape[0]
    dtype = vl_embs.dtype
    actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )
    timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
    return (
        actions,
        timestep,
        vl_embs.to(device=device, dtype=dtype),
        state.to(device=device, dtype=dtype),
        embodiment_id.to(device=device),
    )


def make_groot_static_action_module(
    action_head: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    embodiment_id: torch.Tensor | None,
) -> nn.Module:
    velocity_decoder = action_head.action_decoder
    if embodiment_id is not None:
        velocity_decoder = TRTDynamicCategorySpecificMLP(action_head.action_decoder)
    return StaticActionVelocityStep(
        step_encoder=GrootDiTStepEncoder(action_head, embodiment_id),
        action_expert=action_head.model,
        velocity_decoder=velocity_decoder,
        output_tokens=action_head.config.action_horizon,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)


def make_pi05_static_action_module(core: nn.Module, device: torch.device) -> nn.Module:
    return StaticActionVelocityStep(
        step_encoder=PI05PrefixKVStepEncoder(core),
        action_expert=core.paligemma_with_expert.gemma_expert.model,
        velocity_decoder=core.action_out_proj,
        output_tokens=core.config.chunk_size,
    ).eval().to(device=device)


def make_pi05_action_compile_inputs(
    core: nn.Module,
    *,
    batch_size: int,
    prefix_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    chunk_size = core.config.chunk_size
    action_dim = core.config.max_action_dim
    expert_cfg = core.paligemma_with_expert.gemma_expert.model.config
    dtype = next(core.action_in_proj.parameters()).dtype

    x_t = torch.randn(
        batch_size,
        chunk_size,
        action_dim,
        device=device,
        dtype=dtype,
    )
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    prefix_k = torch.zeros(
        expert_cfg.num_hidden_layers,
        batch_size,
        expert_cfg.num_key_value_heads,
        prefix_len,
        expert_cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    prefix_v = torch.zeros_like(prefix_k)
    prefix_pad_masks = torch.ones(batch_size, prefix_len, dtype=torch.bool, device=device)
    position_ids, attention_mask = make_suffix_position_and_mask(
        core,
        prefix_pad_masks,
        x_t,
        device,
    )
    return (
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    )


def build_groot_diffusion_export_params(
    model: nn.Module,
    *,
    context_embs: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    device: torch.device,
    io: PipelineIOSpec = GROOT_EDGE_IO,
    trt_settings: dict | None = None,
    dtype: torch.dtype = torch.float16,
) -> DiffusionEngineSpec:
    action_head = model.action_head
    context_embs = context_embs.to(device=device, dtype=dtype).contiguous()
    state = state.to(device=device, dtype=dtype).contiguous()
    embodiment_id = embodiment_id.to(device=device).contiguous()

    diffusion_module = make_groot_static_action_module(
        action_head,
        device,
        dtype,
        embodiment_id,
    )
    sample_inputs = make_groot_action_compile_inputs(
        action_head.config.action_horizon,
        action_head.config.action_dim,
        context_embs,
        state,
        embodiment_id,
        device,
    )
    cfg = action_head.config
    return DiffusionEngineSpec(
        diffusion_module=diffusion_module,
        sample_inputs=sample_inputs,
        extra_config={
            "engine_role": "single_action_denoising_step",
            **action_rollout_extra_config(
                io,
                GROOT_ACTION_ROLLOUT,
                num_steps=int(action_head.num_inference_timesteps),
                num_timestep_buckets=int(action_head.num_timestep_buckets),
                action_horizon=int(cfg.action_horizon),
                action_dim=int(cfg.action_dim),
                context_seq_len=int(context_embs.shape[1]),
                context_hidden_size=int(context_embs.shape[2]),
                state_horizon=int(state.shape[1]),
                state_dim=int(state.shape[2]),
            ),
        },
        io=io.action,
        trt_settings=dict(trt_settings or DEFAULT_DIFFUSION_TRT_SETTINGS),
        model_type="action",
    )


def build_smolvla_diffusion_export_params(
    core: nn.Module,
    *,
    prefix: dict,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
    trt_settings: dict | None = None,
    model_type: str = "smolvla",
) -> DiffusionEngineSpec:
    from trt.diffusion import SmolVLAPrefixKVStepEncoder, StaticActionVelocityStep
    from trt.utils import make_smolvla_runner_inputs

    class _SmolVLAActionExpert(nn.Module):
        def __init__(self, model_core):
            super().__init__()
            self.vlm_with_expert = model_core.vlm_with_expert

        def forward(self, **kwargs):
            return self.vlm_with_expert.forward(**kwargs)

    action_module = StaticActionVelocityStep(
        step_encoder=SmolVLAPrefixKVStepEncoder(core),
        action_expert=_SmolVLAActionExpert(core),
        velocity_decoder=core.action_out_proj,
        output_tokens=int(core.config.chunk_size),
    ).eval().to(device)

    text_cfg = core.vlm_with_expert.get_vlm_model().config.text_config
    prefix_k = torch.zeros(
        int(core.vlm_with_expert.num_vlm_layers),
        int(prefix["inputs_embeds"].shape[0]),
        int(text_cfg.num_key_value_heads),
        int(prefix["pad_mask"].shape[1]),
        int(text_cfg.head_dim),
        device=device,
        dtype=torch.float16,
    )
    prefix_v = torch.zeros_like(prefix_k)
    batch_size = int(prefix["pad_mask"].shape[0])
    x_t = torch.randn(
        batch_size,
        core.config.chunk_size,
        core.config.max_action_dim,
        device=device,
        dtype=prefix_k.dtype,
    )
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    sample_inputs = make_smolvla_runner_inputs(
        core,
        prefix["pad_mask"],
        prefix_k,
        prefix_v,
        x_t,
        timestep,
        device,
        edge_llm=True,
    )
    sample_inputs = tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in sample_inputs)

    return DiffusionEngineSpec(
        diffusion_module=action_module,
        sample_inputs=sample_inputs,
        extra_config={
            "engine_role": "single_action_denoising_step",
            **action_rollout_extra_config(
                io,
                PI05_ACTION_ROLLOUT,
                num_steps=int(core.config.num_steps),
                chunk_size=int(core.config.chunk_size),
                max_action_dim=int(core.config.max_action_dim),
                prefix_seq_len=int(prefix["pad_mask"].shape[1]),
                num_layers=int(core.vlm_with_expert.num_vlm_layers),
                num_key_value_heads=int(text_cfg.num_key_value_heads),
                head_dim=int(text_cfg.head_dim),
            ),
        },
        io=io.action,
        trt_settings=dict(trt_settings or DEFAULT_DIFFUSION_TRT_SETTINGS),
        model_type=model_type,
        engine_file="diffusion.engine",
    )


def build_pi05_diffusion_export_params(
    core: nn.Module,
    *,
    batch_size: int,
    prefix_len: int,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
    trt_settings: dict | None = None,
    model_type: str = "action",
) -> DiffusionEngineSpec:
    diffusion_module = make_pi05_static_action_module(core, device)
    sample_inputs = make_pi05_action_compile_inputs(
        core,
        batch_size=batch_size,
        prefix_len=prefix_len,
        device=device,
    )
    expert_cfg = core.paligemma_with_expert.gemma_expert.model.config
    return DiffusionEngineSpec(
        diffusion_module=diffusion_module,
        sample_inputs=sample_inputs,
        extra_config={
            "engine_role": "single_action_denoising_step",
            **action_rollout_extra_config(
                io,
                PI05_ACTION_ROLLOUT,
                num_steps=int(core.config.num_inference_steps),
                chunk_size=int(core.config.chunk_size),
                max_action_dim=int(core.config.max_action_dim),
                prefix_seq_len=int(prefix_len),
                num_layers=int(expert_cfg.num_hidden_layers),
                num_key_value_heads=int(expert_cfg.num_key_value_heads),
                head_dim=int(expert_cfg.head_dim),
            ),
        },
        io=io.action,
        trt_settings=dict(trt_settings or DEFAULT_DIFFUSION_TRT_SETTINGS),
        model_type=model_type,
    )
