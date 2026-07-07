from __future__ import annotations

import torch

from trt.compile import compile_trt_module
from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.executor.models.pi05.load.serialize import SerializedPi05Action
from trt.executor.models.pi05.helpers import make_pi05_suffix_position_and_mask
from trt.io_spec import PI05_ACTION_ROLLOUT, action_rollout_extra_config
from trt.modules.export.diffusion import (
    PI05PrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)
from trt.pipelines.parity import maybe_override_language_kv, parity_initial_actions
from trt.serialize import SerializedTRTEngine


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    inputs = maybe_override_language_kv(ctx, inputs)

    device, dtype = ctx.device, ctx.dtype
    cfg = ctx.model.config

    prefix_k = inputs["tensors"]["prefix_k"].to(device=device, dtype=dtype).contiguous()
    prefix_v = inputs["tensors"]["prefix_v"].to(device=device, dtype=dtype).contiguous()

    prefix_pad_mask = inputs.get("metadata", {}).get("prefix_pad_mask")
    if prefix_pad_mask is None:
        prefix_pad_mask = ctx.inference.action_side.get("prefix_pad_mask")
    if prefix_pad_mask is None:
        raise RuntimeError("PI05 diffusion requires prefix_pad_mask from the language stage")
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

    ref_initial_actions = parity_initial_actions(ctx)
    if ref_initial_actions is not None:
        initial_actions = ref_initial_actions.to(device=device, dtype=dtype).contiguous()
    else:
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
        )

    ctx.inference.noise = initial_actions.detach()

    return {
        "diffusion_module": diffusion_module,
        "diffusion_input": diffusion_input,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix_pad_mask,
        "initial_actions": initial_actions,
        "num_steps": int(cfg.num_inference_steps),
    }


def compile(ctx: EdgeContext, inputs: dict) -> dict:
    trt_engine = compile_trt_module(
        inputs["diffusion_module"],
        inputs["diffusion_input"],
        {**ctx.trt_settings, "use_python_runtime": True},
    )

    return {
        "trt_engine": trt_engine,
    }


def load(ctx: EdgeContext, inputs: dict) -> dict:
    serialized_action = SerializedPi05Action(
        SerializedTRTEngine(ctx.engine_root / "action")
    )
    return {
        "serialized_engine": serialized_action,
    }


def _rollout_actions(
    step_runner,
    *,
    actions: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    core,
    num_steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """PI05 flow-matching Euler loop (dt < 0, continuous float timesteps)."""
    actions = actions.clone().to(dtype=dtype)
    dt = -1.0 / float(num_steps)
    device = actions.device
    batch_size = actions.shape[0]

    for step in range(num_steps):
        time = 1.0 + step * dt
        timestep = torch.full(
            (batch_size,),
            time,
            device=device,
            dtype=torch.float32,
        )
        suffix_position_ids, suffix_attention_mask = make_pi05_suffix_position_and_mask(
            core,
            prefix_pad_mask,
            actions,
            device,
        )
        velocity = step_runner(
            actions,
            timestep,
            prefix_k,
            prefix_v,
            suffix_position_ids,
            suffix_attention_mask,
        )[0].to(dtype=dtype)
        actions = actions + dt * velocity

    return actions


def execute(ctx: EdgeContext, inputs: dict) -> dict:
    match ctx.execution_mode:
        case ExecutionMode.EAGER:
            return _run_eager(ctx, inputs)
        case ExecutionMode.IN_MEMORY:
            return _run_trt(ctx, inputs)
        case ExecutionMode.SERIALIZED:
            return _run_serialized(ctx, inputs)

    raise ValueError(f"unsupported execution mode: {ctx.execution_mode}")


def _run_eager(ctx: EdgeContext, inputs: dict) -> dict:
    with torch.no_grad():
        actions = _rollout_actions(
            inputs["diffusion_module"],
            actions=inputs["initial_actions"],
            prefix_k=inputs["prefix_k"],
            prefix_v=inputs["prefix_v"],
            prefix_pad_mask=inputs["prefix_pad_mask"],
            core=ctx.model,
            num_steps=inputs["num_steps"],
            dtype=ctx.dtype,
        )

    return {
        "tensors": {"actions": actions},
        "metadata": {"backend": "eager"},
    }


def _run_trt(ctx: EdgeContext, inputs: dict) -> dict:
    with torch.no_grad():
        actions = _rollout_actions(
            inputs["trt_engine"],
            actions=inputs["initial_actions"],
            prefix_k=inputs["prefix_k"],
            prefix_v=inputs["prefix_v"],
            prefix_pad_mask=inputs["prefix_pad_mask"],
            core=ctx.model,
            num_steps=inputs["num_steps"],
            dtype=ctx.dtype,
        )

    return {
        "tensors": {"actions": actions},
        "metadata": {"backend": "in_memory_trt"},
    }


def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    module = inputs["serialized_engine"]

    with torch.no_grad():
        actions = _rollout_actions(
            module,
            actions=inputs["initial_actions"],
            prefix_k=inputs["prefix_k"],
            prefix_v=inputs["prefix_v"],
            prefix_pad_mask=inputs["prefix_pad_mask"],
            core=ctx.model,
            num_steps=inputs["num_steps"],
            dtype=ctx.dtype,
        )

    return {
        "tensors": {"actions": actions},
        "metadata": {"backend": "serialized_trt"},
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    ctx.actions = result["tensors"]["actions"]
    return result
