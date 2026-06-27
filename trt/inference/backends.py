"""Inference backends: eager reference, in-memory TRT, serialized engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from trt.action_rollout import sample_actions_raw
from trt.inference.context import InferenceContext, LanguageOutputs, StageHandles
from trt.inference.language_prefill import LanguagePrefillInputs, run_language_prefill
from trt.inference.mode import InferenceBackendKind
from trt.serialize import (
    SerializedGrootAction,
    SerializedGrootActionContext,
    SerializedGrootLanguage,
    SerializedGrootVision,
    SerializedTRTEngine,
)


class InferenceBackend(Protocol):
    kind: InferenceBackendKind

    def run_vision(self, ctx: InferenceContext, pixel_values: torch.Tensor) -> torch.Tensor: ...

    def run_language(
        self,
        ctx: InferenceContext,
        prefill: LanguagePrefillInputs,
    ) -> LanguageOutputs: ...

    def run_action_context(self, ctx: InferenceContext, lm_hidden: torch.Tensor) -> torch.Tensor: ...

    def run_action_rollout(
        self,
        ctx: InferenceContext,
        context_embs: torch.Tensor | None,
        noise: torch.Tensor,
        hooks,
    ) -> torch.Tensor: ...

    def has_action_context(self) -> bool: ...


@dataclass
class EagerBackend:
    kind: InferenceBackendKind = InferenceBackendKind.EAGER


@dataclass
class TrtModuleBackend:
    handles: StageHandles
    kind: InferenceBackendKind = InferenceBackendKind.TRT_MODULE

    def run_vision(self, ctx: InferenceContext, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.handles.vision is None:
            raise RuntimeError("TRT vision module is missing")
        return self.handles.vision(pixel_values.contiguous())

    def run_language(self, ctx: InferenceContext, prefill: LanguagePrefillInputs) -> LanguageOutputs:
        if self.handles.language is None:
            raise RuntimeError("TRT language module is missing")
        return run_language_prefill(self.handles.language, prefill, ctx.io.language)

    def has_action_context(self) -> bool:
        return self.handles.action_context is not None

    def run_action_context(self, ctx: InferenceContext, lm_hidden: torch.Tensor) -> torch.Tensor:
        if self.handles.action_context is None:
            raise RuntimeError("TRT action_context module is missing")
        out = self.handles.action_context(lm_hidden.contiguous())
        if isinstance(out, tuple):
            return out[0]
        return out

    def run_action_rollout(
        self,
        ctx: InferenceContext,
        context_embs: torch.Tensor | None,
        noise: torch.Tensor,
        hooks,
    ) -> torch.Tensor:
        if self.handles.action is None:
            raise RuntimeError("TRT action module is missing")
        rollout_ctx = hooks.build_action_rollout_context(
            ctx,
            noise,
            context_embs=context_embs,
        )
        return sample_actions_raw(
            self.handles.action,
            rollout_ctx,
            hooks.action_adapter(ctx),
        )


@dataclass
class SerializedModuleBackend:
    handles: StageHandles
    kind: InferenceBackendKind = InferenceBackendKind.SERIALIZED

    def run_vision(self, ctx: InferenceContext, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.handles.vision is not None:
            return self.handles.vision(pixel_values)
        if ctx.engine_root is None:
            raise RuntimeError("Serialized vision requires handles.vision or ctx.engine_root")
        runner = SerializedGrootVision(SerializedTRTEngine(ctx.engine_root / "visual"))
        return runner(pixel_values)

    def run_language(self, ctx: InferenceContext, prefill: LanguagePrefillInputs) -> LanguageOutputs:
        if self.handles.language is not None:
            return run_language_prefill(self.handles.language, prefill, ctx.io.language)
        if ctx.engine_root is None:
            raise RuntimeError("Serialized language requires handles.language or ctx.engine_root")
        runner = SerializedGrootLanguage(SerializedTRTEngine(ctx.engine_root / "language"))
        return run_language_prefill(runner, prefill, ctx.io.language)

    def has_action_context(self) -> bool:
        return self.handles.action_context is not None or (
            self.handles.plugin_info.get("action_context_engine") is not None
        )

    def run_action_context(self, ctx: InferenceContext, lm_hidden: torch.Tensor) -> torch.Tensor:
        if self.handles.action_context is not None:
            return self.handles.action_context(lm_hidden.to(dtype=torch.float16).contiguous())
        if ctx.engine_root is None:
            raise RuntimeError("Serialized action_context requires handles or ctx.engine_root")
        runner = SerializedGrootActionContext(SerializedTRTEngine(ctx.engine_root / "action_context"))
        return runner(lm_hidden)

    def run_action_rollout(
        self,
        ctx: InferenceContext,
        context_embs: torch.Tensor | None,
        noise: torch.Tensor,
        hooks,
    ) -> torch.Tensor:
        action = self.handles.action
        if action is None and ctx.engine_root is not None:
            action = SerializedGrootAction(SerializedTRTEngine(ctx.engine_root / "action"))
        if action is None:
            raise RuntimeError("Serialized action module is missing")
        rollout_ctx = hooks.build_action_rollout_context(
            ctx,
            noise,
            context_embs=context_embs,
        )
        return sample_actions_raw(
            action,
            rollout_ctx,
            hooks.action_adapter(ctx),
        )


def stage_handles_from_modules(
    *,
    vision=None,
    language=None,
    action_context=None,
    action=None,
    plugin_info: dict[str, Any] | None = None,
) -> StageHandles:
    return StageHandles(
        vision=vision,
        language=language,
        action_context=action_context,
        action=action,
        plugin_info=dict(plugin_info or {}),
    )
