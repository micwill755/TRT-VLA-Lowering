"""Compile sinks: in-memory modules vs serialized Edge-LLM engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn

from trt.compile import compile_trt_module, save_trt_engine_module
from trt.diffusion import DiffusionEngineSpec, save_action_diffusion_engine_for_edge_llm
from trt.export.context import ComponentBuild
from trt.export.mode import ExportMode
from trt.export.settings import in_memory_settings
from trt.language import (
    LanguageEngineSpec,
    compile_language_trt_with_plugin,
    save_language_engine_for_edge_llm,
)
from trt.modules.export.language import CausalLMExportModule
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.vision import VisionEngineSpec, VisualFixedInput, nchw_to_hwc, save_visual_engine_for_edge_llm


class ExportSink(Protocol):
    mode: ExportMode

    def export_vision(
        self,
        pixel_values: torch.Tensor,
        spec: VisionEngineSpec,
        *,
        engine_dir: Path | None,
        device: torch.device,
    ) -> nn.Module | Path: ...

    def export_language(
        self,
        spec: LanguageEngineSpec,
        *,
        engine_dir: Path | None,
        trace_inputs_embeds: torch.Tensor,
    ) -> nn.Module | Path: ...

    def export_diffusion(
        self,
        spec: DiffusionEngineSpec,
        *,
        engine_dir: Path | None,
    ) -> nn.Module | Path: ...

    def export_component(
        self,
        build: ComponentBuild,
        *,
        engine_dir: Path | None,
        input_names: list[str],
        output_names: list[str],
    ) -> nn.Module | Path: ...


@dataclass
class InMemorySink:
    mode: ExportMode = ExportMode.IN_MEMORY
    vision_trt_settings: dict | None = None
    language_trt_settings: dict | None = None
    default_trt_settings: dict | None = None

    def export_vision(
        self,
        pixel_values: torch.Tensor,
        spec: VisionEngineSpec,
        *,
        engine_dir: Path | None,
        device: torch.device,
    ) -> nn.Module:
        del engine_dir
        pixel_values_nchw = pixel_values.to(device=device, dtype=spec.input_dtype).contiguous()
        images_hwc = nchw_to_hwc(pixel_values_nchw)

        visual = VisualFixedInput(
            vision_model=spec.visual_vision_model,
            projector=spec.projector,
            sample_pixel_values=images_hwc,
            select_layer=spec.select_layer,
            pixel_shuffle=spec.pixel_shuffle,
            downsample_ratio=spec.downsample_ratio,
            force_float32_input=spec.force_float32_input,
            cast_output_to_input_dtype=spec.cast_output_to_input_dtype,
            vision_kwargs=spec.vision_kwargs,
        ).eval().to(device)

        settings = in_memory_settings(spec.trt_settings)
        if self.vision_trt_settings:
            settings.update(self.vision_trt_settings)

        patched = patch_vision_attention(
            spec.patch_vision_model,
            batch_size=spec.patch_batch_size,
            seq_len=spec.patch_seq_len,
            name=spec.patch_name,
            allow_attention_mask=spec.allow_attention_mask,
        )
        try:
            return compile_trt_module(visual, (images_hwc,), settings)
        finally:
            if patched:
                restore_attention(patched)

    def export_language(
        self,
        spec: LanguageEngineSpec,
        *,
        engine_dir: Path | None,
        trace_inputs_embeds: torch.Tensor,
    ) -> nn.Module:
        del engine_dir
        from trt.plugin_utils import patch_language_attention, restore_attention

        settings = in_memory_settings(spec.trt_settings)
        if self.language_trt_settings:
            settings.update(self.language_trt_settings)

        wrapper = CausalLMExportModule(
            spec.decoder,
            spec.lm_head,
            select_layer=spec.select_layer,
        ).eval()
        patched = patch_language_attention(
            spec.decoder,
            hidden_size=spec.hidden_size,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            enable_bidirectional_prefill=spec.enable_bidirectional_prefill,
        )
        try:
            module, _max_seq_len = compile_language_trt_with_plugin(
                wrapper,
                trace_inputs_embeds,
                num_layers=spec.num_layers,
                num_key_value_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim,
                device=trace_inputs_embeds.device,
                settings=settings,
                max_seq_len=spec.max_seq_len,
                dtype=spec.export_dtype,
            )
        finally:
            restore_attention(patched)
        return module

    def export_component(
        self,
        build: ComponentBuild,
        *,
        engine_dir: Path | None,
        input_names: list[str],
        output_names: list[str],
    ) -> nn.Module:
        del engine_dir, input_names, output_names
        settings = in_memory_settings(build.trt_settings or self.default_trt_settings or {})
        return compile_trt_module(build.module, build.sample_inputs, settings)

    def export_diffusion(
        self,
        spec: DiffusionEngineSpec,
        *,
        engine_dir: Path | None,
    ) -> nn.Module:
        del engine_dir
        settings = in_memory_settings(
            spec.trt_settings or self.default_trt_settings or {}
        )
        return compile_trt_module(spec.diffusion_module, spec.sample_inputs, settings)


@dataclass
class SerializedSink:
    mode: ExportMode = ExportMode.SERIALIZED

    def export_vision(
        self,
        pixel_values: torch.Tensor,
        spec: VisionEngineSpec,
        *,
        engine_dir: Path | None,
        device: torch.device,
    ) -> Path:
        save_visual_engine_for_edge_llm(
            pixel_values,
            engine_dir,
            spec,
            device=device,
        )
        return engine_dir / "visual.engine"

    def export_language(
        self,
        spec: LanguageEngineSpec,
        *,
        engine_dir: Path | None,
        trace_inputs_embeds: torch.Tensor,
    ) -> Path:
        del trace_inputs_embeds
        save_language_engine_for_edge_llm(engine_dir, spec)
        return engine_dir / "language.engine"

    def export_diffusion(
        self,
        spec: DiffusionEngineSpec,
        *,
        engine_dir: Path | None,
    ) -> Path:
        save_action_diffusion_engine_for_edge_llm(engine_dir, spec)
        return engine_dir / spec.engine_file

    def export_component(
        self,
        build: ComponentBuild,
        *,
        engine_dir: Path | None,
        input_names: list[str],
        output_names: list[str],
    ) -> Path:
        if engine_dir is None:
            raise ValueError("Serialized component export requires engine_dir")
        save_trt_engine_module(
            build.module,
            build.sample_inputs,
            engine_dir,
            engine_file=build.engine_file,
            model_type=build.model_type,
            component=build.component,
            input_names=input_names,
            output_names=output_names,
            trt_settings=build.trt_settings,
            extra_config=build.extra_config,
        )
        return engine_dir / build.engine_file
