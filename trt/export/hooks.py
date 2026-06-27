"""Model-specific hooks for ``VLAExportPipeline``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trt.diffusion import DiffusionEngineSpec
from trt.export.context import ComponentBuild, ExportContext
from trt.export.sinks import ExportSink
from trt.language import LanguageEngineSpec
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm
from trt.vision import VisionEngineSpec


class VLAExportHooks(ABC):
    """Override only the stages that differ between VLAs."""

    tokenizer: Any | None = None

    @abstractmethod
    def preprocess(self, ctx: ExportContext) -> None:
        """Populate ``ctx.pixel_values``, ``ctx.tokenized``, ``ctx.action_side``."""

    @abstractmethod
    def build_vision_spec(self, ctx: ExportContext) -> VisionEngineSpec:
        ...

    def uses_prefix_kv_action(self) -> bool:
        """True when action rollout consumes LM prefix K/V (PI0.5, SmolVLA)."""
        return bool(self.io.lm_to_action_slots) and self.io.action_context is None

    def compute_image_embs(self, ctx: ExportContext, vision_module: nn.Module) -> torch.Tensor | list[torch.Tensor]:
        """Run in-memory vision and return embedding(s) for language packing."""
        from trt.vision import nchw_to_hwc, run_trt_vision_nchw

        images = ctx.action_side.get("images")
        if images is None:
            return run_trt_vision_nchw(vision_module, ctx.pixel_values.to(device=ctx.device))
        return [
            run_trt_vision_nchw(vision_module, image.to(device=ctx.device))
            for image in images
        ]

    def dummy_image_embs(self, ctx: ExportContext) -> torch.Tensor | list[torch.Tensor]:
        spec = ctx.vis_spec
        if spec is None:
            raise RuntimeError("build_vision_spec must run before dummy_image_embs")
        images = ctx.action_side.get("images")
        if images is not None and spec.image_embed_shape:
            return [
                torch.zeros(
                    *spec.image_embed_shape,
                    device=ctx.device,
                    dtype=spec.input_dtype,
                )
                for _ in images
            ]
        if spec.image_embed_shape:
            return torch.zeros(
                *spec.image_embed_shape,
                device=ctx.device,
                dtype=spec.input_dtype,
            )
        return torch.zeros(
            *spec.image_embed_flat_shape,
            device=ctx.device,
            dtype=spec.input_dtype,
        )

    @abstractmethod
    def pack_language_inputs(self, ctx: ExportContext) -> dict:
        """Return packed LM inputs (``inputs_embeds``, masks, ``position_ids``, ...)."""

    @abstractmethod
    def build_language_spec(self, ctx: ExportContext) -> LanguageEngineSpec:
        ...

    @abstractmethod
    def build_chat_template(self, tokenizer: Any) -> dict[str, Any]:
        """Build ``processed_chat_template.json`` for VitRunner."""

    def save_language_artifacts(self, ctx: ExportContext, language_dir: Path) -> None:
        """Write embedding table, tokenizer assets, and chat template into ``language/``."""
        save_embedding_table(ctx.lang_spec.language_model, language_dir)
        save_tokenizer_for_edge_llm(
            language_dir,
            tokenizer=self.tokenizer,
            chat_template=self.build_chat_template(self.tokenizer),
        )

    def has_action_context(self, ctx: ExportContext) -> bool:
        return ctx.io.action_context is not None

    def build_action_context(self, ctx: ExportContext) -> ComponentBuild | None:
        return None

    @abstractmethod
    def build_diffusion_spec(self, ctx: ExportContext) -> DiffusionEngineSpec:
        ...

    def after_export(self, ctx: ExportContext, sink: ExportSink) -> None:
        """Parity checks, serialized runner smoke, fixture dumps."""

    def finalize_plugin_info(self, ctx: ExportContext) -> dict:
        info: dict = {
            "engine_root": str(ctx.engine_root) if ctx.engine_root else None,
            **ctx.io.to_plugin_info(),
        }
        if ctx.lang_spec is not None:
            info["language_max_seq_len"] = int(ctx.lang_spec.max_seq_len)
        if ctx.language_inputs.get("inputs_embeds") is not None:
            info["language_seq_len"] = int(ctx.language_inputs["inputs_embeds"].shape[1])
        if ctx.context_embs is not None:
            info["context_seq_len"] = int(ctx.context_embs.shape[1])
            info["context_hidden_size"] = int(ctx.context_embs.shape[2])
        if ctx.lm_hidden_states is not None:
            info["lm_hidden_size"] = int(ctx.lm_hidden_states.shape[2])
        state = ctx.action_side.get("state")
        if state is not None:
            info["state_shape"] = list(state.shape)
        embodiment_id = ctx.action_side.get("embodiment_id")
        if embodiment_id is not None:
            info["embodiment_id"] = embodiment_id.detach().cpu().tolist()
        return info
