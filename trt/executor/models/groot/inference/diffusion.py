from __future__ import annotations

import torch

from trt.compile import compile_trt_module
from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext
from trt.executor.models.groot.load.serialize import SerializedGrootAction
from trt.modules.export.diffusion import (
    GrootDiTStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
    TRTDynamicCategorySpecificMLPExportModule,
)
from trt.serialize import SerializedTRTEngine


def preprocess(ctx: EdgeContext, inputs: dict) -> dict:
    device, dtype = ctx.device, ctx.dtype
    action_head = ctx.model.action_head
    cfg = action_head.config

    # upstream action_context post: [B, S, 1536]
    context_embs = inputs["tensors"]["context_embs"]
    context_embs = context_embs.to(device=device, dtype=dtype).contiguous()

    state = inputs["state"].to(device=device, dtype=dtype).contiguous()
    embodiment_id = inputs["embodiment_id"].to(device=device, dtype=torch.long)

    # test_vla.py:432-440
    diffusion_module = StaticActionVelocityStepExportModule(
        step_encoder=GrootDiTStepEncoderExportModule(action_head, embodiment_id),
        action_expert=action_head.model,
        velocity_decoder=TRTDynamicCategorySpecificMLPExportModule(
            action_head.action_decoder
        ),
        output_tokens=cfg.action_horizon,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

    action_horizon = cfg.action_horizon
    action_dim = cfg.action_dim
    batch_size = context_embs.shape[0]  # 1, not vision batch

    # trace sample — test_vla.py:559-582
    step_actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    ).contiguous()
    step_timestep = torch.zeros(batch_size, device=device, dtype=torch.long)

    diffusion_input = (
        step_actions,
        step_timestep,
        context_embs,
        state,
        embodiment_id,
    )

    # seeded initial noise for rollout — test_vla.py:485-504
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

    return {
        "diffusion_module": diffusion_module,
        "diffusion_input": diffusion_input,
        "context_embs": context_embs,
        "state": state,
        "embodiment_id": embodiment_id,
        "initial_actions": initial_actions,
        "num_steps": int(action_head.num_inference_timesteps),
        "num_timestep_buckets": int(action_head.num_timestep_buckets),
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
    serialized_action = SerializedGrootAction(
        SerializedTRTEngine(ctx.engine_root / "action")
    )
    ctx.handles.serialized.action = serialized_action
    return {
        "serialized_engine": serialized_action,
    }


def _rollout_actions(
    step_runner,
    *,
    actions: torch.Tensor,
    context_embs: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    num_steps: int,
    num_timestep_buckets: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Manual Euler loop from test_vla.py:506-545."""
    actions = actions.clone().to(dtype=dtype)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        timestep_bucket = int(t_cont * num_timestep_buckets)

        timestep = torch.full(
            (actions.shape[0],),
            timestep_bucket,
            device=actions.device,
            dtype=torch.long,
        )

        velocity = step_runner(
            actions,
            timestep,
            context_embs,
            state,
            embodiment_id,
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
            context_embs=inputs["context_embs"],
            state=inputs["state"],
            embodiment_id=inputs["embodiment_id"],
            num_steps=inputs["num_steps"],
            num_timestep_buckets=inputs["num_timestep_buckets"],
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
            context_embs=inputs["context_embs"],
            state=inputs["state"],
            embodiment_id=inputs["embodiment_id"],
            num_steps=inputs["num_steps"],
            num_timestep_buckets=inputs["num_timestep_buckets"],
            dtype=ctx.dtype,
        )

    return {
        "tensors": {"actions": actions},
        "metadata": {"backend": "in_memory_trt"},
    }


def _run_serialized(ctx: EdgeContext, inputs: dict) -> dict:
    module = inputs.get("serialized_engine") or ctx.handles.serialized.action
    if module is None:
        raise RuntimeError("serialized TRT backend missing action module")

    with torch.no_grad():
        actions = _rollout_actions(
            module,
            actions=inputs["initial_actions"],
            context_embs=inputs["context_embs"],
            state=inputs["state"],
            embodiment_id=inputs["embodiment_id"],
            num_steps=inputs["num_steps"],
            num_timestep_buckets=inputs["num_timestep_buckets"],
            dtype=ctx.dtype,
        )

    return {
        "tensors": {"actions": actions},
        "metadata": {"backend": "serialized_trt"},
    }


def postprocess(ctx: EdgeContext, result: dict) -> dict:
    ctx.actions = result["tensors"]["actions"]
    return result
