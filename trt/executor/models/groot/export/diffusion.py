from __future__ import annotations

import torch

from trt.compile import save_trt_engine_module
from trt.context import EdgeContext
from trt.modules.export.diffusion import (
    GrootDiTStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
    TRTDynamicCategorySpecificMLPExportModule,
)

def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    device, dtype = ctx.device, ctx.dtype
    action_head = ctx.model.action_head
    cfg = action_head.config

    # upstream action_context output: [B, S_ctx, H_ctx]
    context_embs = inputs["tensors"]["context_embs"]
    context_embs = context_embs.to(device=device, dtype=dtype).contiguous()

    state = inputs["state"].to(device=device, dtype=dtype).contiguous()
    embodiment_id = inputs["embodiment_id"].to(device=device, dtype=torch.long)

    diffusion_module = StaticActionVelocityStepExportModule(
        step_encoder=GrootDiTStepEncoderExportModule(action_head, embodiment_id),
        action_expert=action_head.model,
        velocity_decoder=TRTDynamicCategorySpecificMLPExportModule(
            action_head.action_decoder
        ),
        output_tokens=cfg.action_horizon,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

    batch_size = int(context_embs.shape[0])
    action_horizon = int(cfg.action_horizon)
    action_dim = int(cfg.action_dim)

    step_actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    ).contiguous()

    step_timestep = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.long,
    )

    diffusion_input = (
        step_actions,
        step_timestep,
        context_embs,
        state,
        embodiment_id,
    )

    seed = getattr(ctx.inference, "seed", 42)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    initial_actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()

    return {
        "diffusion_module": diffusion_module,
        "diffusion_input": diffusion_input,
        "context_embs": context_embs,
        "state": state,
        "embodiment_id": embodiment_id,
        "initial_actions": initial_actions,
        "batch_size": batch_size,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "context_seq_len": int(context_embs.shape[1]),
        "context_hidden_size": int(context_embs.shape[2]),
        "state_horizon": int(state.shape[1]),
        "state_dim": int(state.shape[2]),
        "num_steps": int(action_head.num_inference_timesteps),
        "num_timestep_buckets": int(action_head.num_timestep_buckets),
        "language_inputs": inputs.get("metadata", {}).get("language_inputs"),
    }

def export(ctx: EdgeContext, inputs: dict) -> dict:
    diffusion_module = inputs["diffusion_module"]
    diffusion_input = inputs["diffusion_input"]

    engine_path = save_trt_engine_module(
        diffusion_module,
        diffusion_input,
        ctx.engine_root / "action",
        engine_file="action.engine",
        model_type="action",
        component="diffusion",
        input_names=[
            "actions",
            "timestep",
            "context_embs",
            "state",
            "embodiment_id",
        ],
        output_names=["velocity"],
        extra_config={
            "engine_role": "single_action_denoising_step",
            "noise_input_name": "actions",
            "timestep_schedule": "discrete_buckets",
            "rollout_dt_sign": 1,
            "num_inference_timesteps": int(inputs["num_steps"]),
            "num_timestep_buckets": int(inputs["num_timestep_buckets"]),
            "action_horizon": int(inputs["action_horizon"]),
            "action_dim": int(inputs["action_dim"]),
            "context_seq_len": int(inputs["context_seq_len"]),
            "context_hidden_size": int(inputs["context_hidden_size"]),
            "state_horizon": int(inputs["state_horizon"]),
            "state_dim": int(inputs["state_dim"]),
        },
        trt_settings=ctx.trt_settings,
    )

    # Dummy final actions for export metadata/stage output. Export writes a
    # single-step velocity engine; runtime performs the denoising rollout.
    actions = torch.zeros(
        inputs["batch_size"],
        inputs["action_horizon"],
        inputs["action_dim"],
        device=ctx.device,
        dtype=ctx.dtype,
    )

    return {
        "engine_path": engine_path,
        "tensors": {
            "actions": actions,
        },
        "metadata": {
            "language_inputs": inputs.get("language_inputs"),
            "context_seq_len": inputs["context_seq_len"],
            "context_hidden_size": inputs["context_hidden_size"],
            "action_horizon": inputs["action_horizon"],
            "action_dim": inputs["action_dim"],
            "num_steps": inputs["num_steps"],
            "num_timestep_buckets": inputs["num_timestep_buckets"],
        },
    }

def postprocess(ctx: EdgeContext, result: dict) -> dict:
    return result