"""Model-specific hooks for ``VLAInferencePipeline``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from trt.inference.context import InferenceContext, LanguageOutputs
from trt.inference.language_prefill import LanguagePrefillInputs


class VLAInferenceHooks(ABC):
    @abstractmethod
    def preprocess(self, ctx: InferenceContext) -> None:
        """Populate ``ctx.pixel_values``, ``ctx.tokenized``, ``ctx.action_side``."""

    @abstractmethod
    def pack_language_inputs(self, ctx: InferenceContext) -> dict:
        ...

    @abstractmethod
    def language_model_for_prefill(self, ctx: InferenceContext) -> nn.Module:
        ...

    @abstractmethod
    def language_prefill_scalars(self, ctx: InferenceContext) -> dict[str, int]:
        """Return num_layers, num_key_value_heads, head_dim, max_seq_len."""

    @abstractmethod
    def make_rollout_noise(self, ctx: InferenceContext, context_embs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def action_adapter(self, ctx: InferenceContext):
        ...

    def uses_prefix_kv_action(self, ctx: InferenceContext) -> bool:
        return bool(ctx.io.lm_to_action_slots) and ctx.io.action_context is None

    def build_action_rollout_context(
        self,
        ctx: InferenceContext,
        noise: torch.Tensor,
        *,
        context_embs: torch.Tensor | None = None,
    ):
        from trt.action_rollout import ActionRolloutContext

        return ActionRolloutContext(
            noise=noise,
            device=ctx.device,
            context_embs=context_embs,
            state=ctx.action_side.get("state"),
            embodiment_id=ctx.action_side.get("embodiment_id"),
        )

    def run_eager_e2e(self, ctx: InferenceContext) -> None:
        """Optional fused eager path (HF LM + context in one shot)."""
        raise NotImplementedError

    def eager_vision(self, ctx: InferenceContext, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eager_lm_hidden(self, ctx: InferenceContext, language_inputs: dict) -> torch.Tensor:
        raise NotImplementedError

    def eager_context_embs(self, ctx: InferenceContext, language_inputs: dict) -> torch.Tensor:
        raise NotImplementedError

    def eager_action_module(self, ctx: InferenceContext) -> nn.Module:
        raise NotImplementedError

    def compare_stage_parity(
        self,
        ctx: InferenceContext,
        *,
        reference_lm: LanguageOutputs | None = None,
    ) -> None:
        """Optional per-stage diff after candidate run."""

    def finalize_extras(self, ctx: InferenceContext) -> dict[str, Any]:
        return dict(ctx.extras)
