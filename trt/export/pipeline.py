"""Shared VLA export orchestrator."""

from __future__ import annotations

from pathlib import Path

import torch

from trt.export.context import ExportContext, PipelineResult
from trt.export.hooks import VLAExportHooks
from trt.export.mode import ExportMode
from trt.export.sinks import ExportSink, InMemorySink, SerializedSink
from trt.io_spec import PipelineIOSpec
from trt.plugin_utils import load_plugins_for_trt


class VLAExportPipeline:
    """Run vision -> language -> [action_context] -> action for any VLA."""

    def __init__(self, hooks: VLAExportHooks, *, io: PipelineIOSpec):
        self.hooks = hooks
        self.io = io

    def run(
        self,
        model,
        policy,
        device: torch.device,
        model_inputs: dict,
        *,
        mode: ExportMode,
        engine_root: str | Path | None = None,
        seed: int = 42,
        max_seq_len: int | None = None,
        accuracy_check: bool = True,
        sink: ExportSink | None = None,
    ) -> PipelineResult:
        if sink is None:
            sink = SerializedSink() if mode is ExportMode.SERIALIZED else InMemorySink()

        ctx = ExportContext(
            model=model,
            policy=policy,
            device=device,
            model_inputs=model_inputs,
            io=self.io,
            engine_root=Path(engine_root) if engine_root else None,
            seed=seed,
            max_seq_len=max_seq_len,
            accuracy_check=accuracy_check,
        )

        self.hooks.preprocess(ctx)

        print("compiling vision")
        ctx.vis_spec = self.hooks.build_vision_spec(ctx)
        vision_dir = ctx.engine_subdir("visual")
        ctx.handles["vision"] = sink.export_vision(
            ctx.pixel_values,
            ctx.vis_spec,
            engine_dir=vision_dir,
            device=ctx.device,
        )

        if sink.mode is ExportMode.IN_MEMORY:
            with torch.no_grad():
                ctx.image_embs = self.hooks.compute_image_embs(ctx, ctx.handles["vision"])
        else:
            ctx.image_embs = self.hooks.dummy_image_embs(ctx)

        ctx.language_inputs = self.hooks.pack_language_inputs(ctx)
        ctx.lang_spec = self.hooks.build_language_spec(ctx)
        trace_embeds = ctx.language_inputs["inputs_embeds"]

        language_dir = ctx.engine_subdir("language")
        if language_dir is not None and self.hooks.tokenizer is not None:
            print("saving language sidecars")
            self.hooks.save_language_artifacts(ctx, language_dir)

        print("compiling language.engine")
        custom_lm = self.hooks.compile_language_in_memory(ctx, sink)
        if custom_lm is not None and sink.mode is ExportMode.IN_MEMORY:
            ctx.handles["language"] = custom_lm
        else:
            ctx.handles["language"] = sink.export_language(
                ctx.lang_spec,
                engine_dir=language_dir,
                trace_inputs_embeds=trace_embeds,
            )

        if sink.mode is ExportMode.SERIALIZED:
            ctx.lm_hidden_states = torch.zeros(
                ctx.lang_spec.batch_size,
                ctx.lang_spec.max_seq_len,
                ctx.lang_spec.hidden_size,
                device=ctx.device,
                dtype=ctx.lang_spec.export_dtype,
            )
        elif ctx.lm_hidden_states is None:
            lm_out = self._run_in_memory_language(ctx)
            ctx.lm_hidden_states = lm_out.lm_hidden_states
            if self.hooks.uses_prefix_kv_action():
                ctx.action_side["prefix_k"] = lm_out.prefix_k
                ctx.action_side["prefix_v"] = lm_out.prefix_v
                pad_mask = ctx.language_inputs.get("pad_mask")
                if pad_mask is not None:
                    ctx.action_side["prefix_pad_mask"] = pad_mask

        if self.hooks.has_action_context(ctx):
            print("compiling action context")
            action_context_build = self.hooks.build_action_context(ctx)
            if action_context_build is None:
                raise RuntimeError("action_context IO spec set but build_action_context returned None")
            ac_dir = ctx.engine_subdir("action_context")
            ctx.handles["action_context"] = sink.export_component(
                action_context_build,
                engine_dir=ac_dir,
                input_names=list(self.io.action_context.input_names),
                output_names=list(self.io.action_context.output_names),
            )
            if sink.mode is ExportMode.SERIALIZED:
                ctx.context_embs = torch.zeros(
                    ctx.lang_spec.batch_size,
                    ctx.lang_spec.max_seq_len,
                    int(action_context_build.extra_config.get("context_hidden_size", ctx.lang_spec.hidden_size)),
                    device=ctx.device,
                    dtype=ctx.lang_spec.export_dtype,
                )
            elif ctx.context_embs is None:
                with torch.no_grad():
                    ctx.context_embs = action_context_build.module(*action_context_build.sample_inputs)

        print("compiling action")
        ctx.diffusion_spec = self.hooks.build_diffusion_spec(ctx)
        action_dir = ctx.engine_subdir("action")
        ctx.handles["action"] = sink.export_diffusion(
            ctx.diffusion_spec,
            engine_dir=action_dir,
        )

        if ctx.context_embs is None and ctx.diffusion_spec.sample_inputs:
            ctx.context_embs = ctx.diffusion_spec.sample_inputs[2]

        if accuracy_check:
            self.hooks.after_export(ctx, sink)

        return PipelineResult(handles=ctx.handles)

    def _run_in_memory_language(self, ctx: ExportContext):
        """Default in-memory LM forward parsed into ``LanguageOutputs``."""
        from trt.inference.context import LanguageOutputs
        from trt.inference.language_prefill import build_language_prefill_inputs, run_language_prefill

        spec = ctx.lang_spec
        prefill = build_language_prefill_inputs(
            ctx.language_inputs,
            language_model=spec.language_model,
            num_layers=spec.num_layers,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            max_seq_len=spec.max_seq_len,
            device=ctx.device,
            dtype=spec.export_dtype,
        )
        with torch.no_grad():
            return run_language_prefill(ctx.handles["language"], prefill, ctx.io.language)
