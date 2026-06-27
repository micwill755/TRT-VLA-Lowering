"""Shared VLA inference orchestrator."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from trt.inference.backends import EagerBackend, InferenceBackend
from trt.inference.context import InferenceContext, InferenceResult, LanguageOutputs
from trt.inference.hooks import VLAInferenceHooks
from trt.inference.language_prefill import build_language_prefill_inputs
from trt.inference.mode import InferenceMode
from trt.io_spec import PipelineIOSpec
from trt.measure import tensor_error_metrics


class VLAInferencePipeline:
    def __init__(self, hooks: VLAInferenceHooks, *, io: PipelineIOSpec):
        self.hooks = hooks
        self.io = io

    @torch.no_grad()
    def run(
        self,
        model,
        policy,
        device: torch.device,
        model_inputs: dict,
        backend: InferenceBackend,
        *,
        mode: InferenceMode = InferenceMode.E2E,
        reference: InferenceBackend | None = None,
        seed: int = 42,
        vision_module=None,
        plugin_info: dict | None = None,
        engine_root: str | Path | None = None,
    ) -> InferenceResult:
        ctx = InferenceContext(
            model=model,
            policy=policy,
            device=device,
            model_inputs=model_inputs,
            io=self.io,
            seed=seed,
            vision_module=vision_module,
            engine_root=Path(engine_root) if engine_root else None,
        )
        if plugin_info:
            ctx.plugin_info.update(plugin_info)
        elif getattr(backend, "handles", None) and backend.handles.plugin_info:
            ctx.plugin_info.update(backend.handles.plugin_info)

        self._seed(ctx)
        t0 = time.perf_counter()

        if isinstance(backend, EagerBackend):
            self.hooks.preprocess(ctx)
            self.hooks.run_eager_e2e(ctx)
            elapsed = time.perf_counter() - t0
            ctx.extras.update(self.hooks.finalize_extras(ctx))
            return InferenceResult(
                actions=ctx.actions,
                extras=ctx.extras,
                elapsed_s=elapsed,
                ctx=ctx,
            )

        self.hooks.preprocess(ctx)

        t_stage = time.perf_counter()
        run_vision_embs = getattr(self.hooks, "run_vision_embs", None)
        if run_vision_embs is not None:
            ctx.image_embs = run_vision_embs(ctx, backend)
        elif ctx.vision_module is not None:
            ctx.image_embs = ctx.vision_module(ctx.pixel_values.contiguous())
        else:
            ctx.image_embs = backend.run_vision(ctx, ctx.pixel_values)
        ctx.stage_ms["vision"] = (time.perf_counter() - t_stage) * 1000

        ctx.language_inputs = self.hooks.pack_language_inputs(ctx)
        prefill = build_language_prefill_inputs(
            ctx.language_inputs,
            language_model=self.hooks.language_model_for_prefill(ctx),
            device=device,
            **self.hooks.language_prefill_scalars(ctx),
        )

        t_stage = time.perf_counter()
        ctx.lm = backend.run_language(ctx, prefill)
        ctx.stage_ms["language"] = (time.perf_counter() - t_stage) * 1000

        lm_hidden = ctx.lm.lm_hidden_states
        if lm_hidden is None and not self.hooks.uses_prefix_kv_action(ctx):
            raise RuntimeError("language backend did not produce lm_hidden_states")

        if self.hooks.uses_prefix_kv_action(ctx):
            if ctx.lm.prefix_k is None or ctx.lm.prefix_v is None:
                raise RuntimeError("prefix-KV language backend did not produce prefix_k/prefix_v")
            ctx.action_side["prefix_k"] = ctx.lm.prefix_k
            ctx.action_side["prefix_v"] = ctx.lm.prefix_v
            pad_mask = ctx.language_inputs.get("pad_mask")
            if pad_mask is not None:
                ctx.action_side["prefix_pad_mask"] = pad_mask
            ctx.noise = self.hooks.make_rollout_noise(ctx, lm_hidden or ctx.lm.prefix_k)

            t_stage = time.perf_counter()
            ctx.actions = backend.run_action_rollout(ctx, None, ctx.noise, self.hooks)
            ctx.stage_ms["action"] = (time.perf_counter() - t_stage) * 1000
        else:
            if lm_hidden is None:
                raise RuntimeError("language backend did not produce lm_hidden_states")

            if backend.has_action_context():
                t_stage = time.perf_counter()
                ctx.context_embs = backend.run_action_context(ctx, lm_hidden)
                ctx.stage_ms["action_context"] = (time.perf_counter() - t_stage) * 1000
            else:
                ctx.context_embs = lm_hidden

            ctx.context_embs = ctx.context_embs.to(device=device, dtype=torch.float16).contiguous()
            ctx.noise = self.hooks.make_rollout_noise(ctx, ctx.context_embs)

            t_stage = time.perf_counter()
            ctx.actions = backend.run_action_rollout(ctx, ctx.context_embs, ctx.noise, self.hooks)
            ctx.stage_ms["action"] = (time.perf_counter() - t_stage) * 1000

        elapsed = time.perf_counter() - t0

        if mode is InferenceMode.STAGE_PARITY:
            self._run_stage_parity(ctx, backend, reference)

        ctx.extras.update(
            {
                "noise": ctx.noise,
                "visual_embeds": ctx.image_embs,
                "context_embs": ctx.context_embs,
                "state": ctx.action_side.get("state"),
                "embodiment_id": ctx.action_side.get("embodiment_id"),
                "stage_ms": dict(ctx.stage_ms),
            }
        )
        ctx.extras.update(self.hooks.finalize_extras(ctx))

        return InferenceResult(
            actions=ctx.actions,
            extras=ctx.extras,
            elapsed_s=elapsed,
            ctx=ctx,
        )

    def _run_stage_parity(
        self,
        ctx: InferenceContext,
        backend: InferenceBackend,
        reference: InferenceBackend | None,
    ) -> None:
        self.hooks.compare_stage_parity(ctx)
        if reference is None:
            return

        ref_ctx = InferenceContext(
            model=ctx.model,
            policy=ctx.policy,
            device=ctx.device,
            model_inputs=ctx.model_inputs,
            io=ctx.io,
            seed=ctx.seed,
            plugin_info=dict(ctx.plugin_info),
            vision_module=ctx.vision_module,
            engine_root=ctx.engine_root,
        )
        self.hooks.preprocess(ref_ctx)
        ref_ctx.language_inputs = dict(ctx.language_inputs)

        ref_image = self.hooks.eager_vision(ref_ctx, ref_ctx.pixel_values)
        tensor_error_metrics(
            "vision",
            ctx.image_embs.to(device=ctx.device, dtype=torch.float16),
            ref_image.to(device=ctx.device, dtype=torch.float16),
        )

        ref_lm_hidden = self.hooks.eager_lm_hidden(ref_ctx, ref_ctx.language_inputs)
        tensor_error_metrics(
            "language lm_hidden_states",
            ctx.lm.lm_hidden_states.to(device=ctx.device, dtype=torch.float16),
            ref_lm_hidden.to(device=ctx.device, dtype=torch.float16),
        )

        ref_context = self.hooks.eager_context_embs(ref_ctx, ref_ctx.language_inputs)
        tensor_error_metrics(
            "action_context vl_embs",
            ctx.context_embs.to(device=ctx.device, dtype=torch.float16),
            ref_context.to(device=ctx.device, dtype=torch.float16),
        )

    @staticmethod
    def _seed(ctx: InferenceContext) -> None:
        torch.manual_seed(ctx.seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.seed)
