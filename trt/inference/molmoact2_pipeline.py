"""MolmoAct2 inference orchestrator (fused backbone + action flow)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from trt.inference.hooks import VLAInferenceHooks
from trt.inference.mode import InferenceMode
from trt.io_spec import PipelineIOSpec


def run_molmoact2_backbone(
    runner,
    language_inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mask = language_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(language_inputs["input_ids"], dtype=torch.bool)
    outputs = runner(
        language_inputs["input_ids"],
        language_inputs["pixel_values"],
        language_inputs["image_token_pooling"],
        language_inputs["image_grids"],
        language_inputs["image_num_crops"],
        attention_mask,
    )
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    raise RuntimeError("MolmoAct2 backbone runner returned unexpected output type")


def encoder_attention_mask_for_rollout(
    policy,
    language_inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    attention_mask = language_inputs.get("attention_mask")
    encoder_attention_mask = policy._encoder_attention_mask_for_action_expert(
        input_ids=language_inputs["input_ids"],
        attention_mask=attention_mask,
    )
    if encoder_attention_mask is None:
        if attention_mask is None:
            encoder_attention_mask = torch.ones_like(
                language_inputs["input_ids"],
                device=device,
                dtype=torch.bool,
            )
        else:
            encoder_attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    return encoder_attention_mask.to(device=device, dtype=dtype).contiguous()


class MolmoAct2InferencePipeline:
    """Backbone KV prefill -> continuous flow action rollout (no vision stage)."""

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
        seed: int = 42,
        plugin_info: dict | None = None,
        engine_root: str | Path | None = None,
    ) -> InferenceResult:
        del model, mode
        ctx = InferenceContext(
            model=model,
            policy=policy,
            device=device,
            model_inputs=model_inputs,
            io=self.io,
            seed=seed,
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
        dtype = ctx.action_side["export_dtype"]
        ctx.language_inputs = self.hooks.pack_language_inputs(ctx)

        t_stage = time.perf_counter()
        backbone_runner = getattr(backend, "handles", None)
        backbone_runner = backbone_runner.language if backbone_runner is not None else None
        if backbone_runner is None:
            raise RuntimeError("MolmoAct2 TRT backend is missing language/backbone handle")
        encoder_k, encoder_v = run_molmoact2_backbone(backbone_runner, ctx.language_inputs)
        ctx.stage_ms["language"] = (time.perf_counter() - t_stage) * 1000

        ctx.action_side["encoder_k"] = encoder_k
        ctx.action_side["encoder_v"] = encoder_v
        ctx.action_side["encoder_attention_mask"] = encoder_attention_mask_for_rollout(
            policy,
            ctx.language_inputs,
            device=device,
            dtype=dtype,
        )

        ctx.noise = self.hooks.make_rollout_noise(ctx, encoder_k)

        t_stage = time.perf_counter()
        ctx.actions = backend.run_action_rollout(ctx, None, ctx.noise, self.hooks)
        ctx.stage_ms["action"] = (time.perf_counter() - t_stage) * 1000

        elapsed = time.perf_counter() - t0
        ctx.extras.update(
            {
                "noise": ctx.noise,
                "encoder_k": encoder_k,
                "encoder_v": encoder_v,
                "encoder_attention_mask": ctx.action_side["encoder_attention_mask"],
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

    @staticmethod
    def _seed(ctx: InferenceContext) -> None:
        torch.manual_seed(ctx.seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.seed)
