"""GR00T action export hooks: flow-matching denoising step → TRT action engine.

Stage 3 (final) in the GROOT export pipeline. ``glue.action_context_to_action`` wires
stage-2 ``context_embs`` plus preprocess ``state`` / ``embodiment_id`` into
``stage_inputs``; ``plan_export`` builds one denoising-step ``ExportPlan``;
``compile`` writes ``action/action.engine``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.plan import ExportPlan
from trt.io_spec import GROOT_ACTION_ROLLOUT, GROOT_EDGE_IO, action_rollout_extra_config
from trt.modules.export.diffusion import (
    DEFAULT_DIFFUSION_TRT_SETTINGS,
    GrootDiTStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
    TRTDynamicCategorySpecificMLPExportModule,
)
from trt.runner.base import StageContext


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
    """Build the GR00T action TRT export plan (single flow-matching denoising step).

    ``stage_inputs`` come from ``glue.action_context_to_action`` (upstream stage 2).

    Shape flow (typical libero export, B=1)::

        context_embs   [B, T_ctx, H_ctx]  projected LM context (vl_embs); trace uses zeros
        state          [B, 1, D_state]    packed proprio (D_state = max_state_dim, e.g. 64)
        embodiment_id  [B]               embodiment category index
        actions        [B, T_act, D_act] noisy action trajectory (random for trace)
        timestep       [B]               discrete bucket index (0 for trace)
        velocity out   [B, T_act, D_act]  one Euler denoising step output

    ``T_act`` / ``D_act`` = ``action_horizon`` / ``action_dim`` from ``action_head.config``.
    ``T_ctx`` / ``H_ctx`` = sequence length and hidden size after action_context (stage 2).

    At runtime C++ rolls ``num_inference_timesteps`` calls to this engine (see
    ``extra_config``), updating ``actions`` each step — export traces **one** step only.
    """
    dtype = torch.float16

    # --- 1. Stage inputs from glue + preprocess --------------------------------
    # context_embs: dummy zeros at export time; real vl_embs at inference.
    context_embs = stage_inputs["context_embs"].to(
        device=ctx.device,
        dtype=dtype,
    ).contiguous()  # [B, T_ctx, H_ctx]
    # state / embodiment_id: from preprocess export_state["action_side"] via glue.
    state = stage_inputs["state"].to(device=ctx.device, dtype=dtype).contiguous()  # [B, 1, D_state]
    embodiment_id = stage_inputs["embodiment_id"].to(device=ctx.device).contiguous()  # [B]

    action_head = ctx.model.action_head
    cfg = action_head.config
    batch_size = context_embs.shape[0]  # B

    # --- 2. Trace target: one denoising step -----------------------------------
    # StaticActionVelocityStepExportModule composes:
    #   GrootDiTStepEncoderExportModule  — encode (actions, t, context, state, emb_id)
    #   action_head.model (DiT)          — cross-attend to context_embs
    #   action_decoder                   — project hidden → velocity
    #
    # TRTDynamicCategorySpecificMLPExportModule wraps the embodiment-conditioned
    # decoder when embodiment_id is a runtime tensor (GR00T default).
    velocity_decoder = action_head.action_decoder
    if embodiment_id is not None:
        velocity_decoder = TRTDynamicCategorySpecificMLPExportModule(
            action_head.action_decoder
        )
    diffusion_module = StaticActionVelocityStepExportModule(
        step_encoder=GrootDiTStepEncoderExportModule(action_head, embodiment_id),
        action_expert=action_head.model,
        velocity_decoder=velocity_decoder,
        output_tokens=action_head.config.action_horizon,
        cast_hidden_fp32=False,
    ).eval().to(device=ctx.device, dtype=dtype)

    # --- 3. Sample inputs matching GROOT_EDGE_IO.action binding names ----------
    # Order: actions, timestep, context_embs, state, embodiment_id
    sample_inputs = (
        torch.randn(
            batch_size,
            cfg.action_horizon,
            cfg.action_dim,
            device=ctx.device,
            dtype=dtype,
        ),  # actions [B, T_act, D_act]
        torch.zeros(batch_size, device=ctx.device, dtype=torch.long),  # timestep [B]
        context_embs,
        state,
        embodiment_id,
    )

    # --- 4. C++ rollout metadata (merged into action/config.json) ---------------
    # discrete_buckets schedule + num_inference_timesteps for multi-step denoising loop.
    extra_config = {
        "engine_role": "single_action_denoising_step",
        **action_rollout_extra_config(
            GROOT_EDGE_IO,
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
    }

    # --- 5. ExportPlan — consumed by ExportRunner + compile() hook -------------
    return ExportPlan(
        module=diffusion_module,
        sample_inputs=sample_inputs,
        input_names=tuple(GROOT_EDGE_IO.action.input_names),
        # ("actions", "timestep", "context_embs", "state", "embodiment_id")
        output_names=tuple(GROOT_EDGE_IO.action.output_names),  # ("velocity",)
        engine_dir=ctx.engine_root / "action",
        engine_file="action.engine",
        model_type="action",
        component="diffusion",
        trt_settings=dict(DEFAULT_DIFFUSION_TRT_SETTINGS),
        cleanup_modules=(diffusion_module,),
        args={
            "extra_config": extra_config,
            "context_seq_len": int(context_embs.shape[1]),
            "context_hidden_size": int(context_embs.shape[2]),
            "language_inputs": stage_inputs.get("language_inputs"),
        },
    )


def compile(plan: ExportPlan) -> Path:
    """Trace ``plan.module`` and write ``action/action.engine`` + ``config.json``.

    Single-step flow-matching module; no attention patching (unlike vision/language).
    ``extra_config`` carries rollout schedule for the C++ denoising loop at inference.
    """
    return save_trt_engine_module(
        plan.module,
        plan.sample_inputs,
        plan.engine_dir,
        engine_file=plan.engine_file,
        model_type=plan.model_type or "action",
        component=plan.component or "diffusion",
        input_names=list(plan.input_names),
        output_names=list(plan.output_names),
        example_output=None,
        extra_config=plan.args["extra_config"],
        trt_settings=plan.trt_settings,
    )


def metadata(ctx: StageContext, plan: ExportPlan) -> dict:
    """Return stage-3 artifacts on ``StageResult.metadata`` (final pipeline stage)."""
    del ctx
    args = plan.args
    return {
        "language_inputs": args.get("language_inputs"),
        "context_seq_len": args["context_seq_len"],
        "context_hidden_size": args["context_hidden_size"],
    }
