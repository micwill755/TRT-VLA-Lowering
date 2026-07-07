from __future__ import annotations

import torch

from trt.compile import save_trt_engine_module
from trt.context import EdgeContext
from trt.executor.models.pi05.helpers import make_pi05_suffix_position_and_mask
from trt.io_spec import PI05_ACTION_ROLLOUT, PI05_EDGE_IO, action_rollout_extra_config
from trt.modules.export.diffusion import (
    PI05PrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    device, dtype = ctx.device, ctx.dtype
    cfg = ctx.model.config

    prefix_k = inputs["tensors"]["prefix_k"].to(device=device, dtype=dtype).contiguous()
    prefix_v = inputs["tensors"]["prefix_v"].to(device=device, dtype=dtype).contiguous()

    prefix_pad_mask = inputs.get("metadata", {}).get("prefix_pad_mask")
    if prefix_pad_mask is None:
        raise RuntimeError("PI05 diffusion export requires prefix_pad_mask from language stage")
    prefix_pad_mask = prefix_pad_mask.to(device=device)

    diffusion_module = StaticActionVelocityStepExportModule(
        step_encoder=PI05PrefixKVStepEncoderExportModule(ctx.model),
        action_expert=ctx.model.paligemma_with_expert.gemma_expert.model,
        velocity_decoder=ctx.model.action_out_proj,
        output_tokens=cfg.chunk_size,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

    batch_size = int(prefix_k.shape[1])
    action_horizon = int(cfg.chunk_size)
    action_dim = int(cfg.max_action_dim)

    step_actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    ).contiguous()
    step_timestep = torch.full(
        (batch_size,),
        1.0,
        device=device,
        dtype=torch.float32,
    )

    suffix_position_ids, suffix_attention_mask = make_pi05_suffix_position_and_mask(
        ctx.model,
        prefix_pad_mask,
        step_actions,
        device,
    )

    diffusion_input = (
        step_actions,
        step_timestep,
        prefix_k,
        prefix_v,
        suffix_position_ids,
        suffix_attention_mask,
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
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix_pad_mask,
        "initial_actions": initial_actions,
        "batch_size": batch_size,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "prefix_seq_len": int(prefix_pad_mask.shape[1]),
        "num_steps": int(cfg.num_inference_steps),
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
        input_names=list(PI05_EDGE_IO.action.input_names),
        output_names=list(PI05_EDGE_IO.action.output_names),
        extra_config=action_rollout_extra_config(
            PI05_EDGE_IO,
            PI05_ACTION_ROLLOUT,
            num_steps=int(inputs["num_steps"]),
            action_horizon=int(inputs["action_horizon"]),
            action_dim=int(inputs["action_dim"]),
            prefix_seq_len=int(inputs["prefix_seq_len"]),
        ),
        trt_settings=ctx.trt_settings,
    )

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
            "action_horizon": inputs["action_horizon"],
            "action_dim": inputs["action_dim"],
            "num_steps": inputs["num_steps"],
            "prefix_seq_len": inputs["prefix_seq_len"],
        },
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    return result
