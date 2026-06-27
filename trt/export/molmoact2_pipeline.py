"""MolmoAct2 export orchestrator (fused backbone + action flow)."""

from __future__ import annotations

from pathlib import Path

import torch

from trt.export.context import ExportContext, PipelineResult
from trt.export.hooks import VLAExportHooks
from trt.export.mode import ExportMode
from trt.export.sinks import ExportSink, InMemorySink, SerializedSink
from trt.io_spec import PipelineIOSpec
from trt.plugin_utils import load_plugins_for_trt


class MolmoAct2ExportPipeline:
    """Backbone KV prefill -> action flow step (no separate VitRunner vision stage)."""

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

        load_plugins_for_trt()
        self.hooks.preprocess(ctx)
        ctx.language_inputs = self.hooks.pack_language_inputs(ctx)

        print("compiling MolmoAct2 backbone (encoder KV)")
        backbone_build = self.hooks.build_backbone_component(ctx)
        language_dir = ctx.engine_subdir("language")
        ctx.handles["language"] = sink.export_component(
            backbone_build,
            engine_dir=language_dir,
            input_names=list(self.io.language.input_names),
            output_names=list(self.io.language.output_names),
        )

        with torch.no_grad():
            encoder_k, encoder_v = backbone_build.module(*backbone_build.sample_inputs)
        ctx.action_side["encoder_k"] = encoder_k
        ctx.action_side["encoder_v"] = encoder_v
        language_inputs = self.hooks.pack_language_inputs(ctx)
        attention_mask = language_inputs.get("attention_mask")
        encoder_attention_mask = ctx.policy._encoder_attention_mask_for_action_expert(
            input_ids=language_inputs["input_ids"],
            attention_mask=attention_mask,
        )
        if encoder_attention_mask is None:
            if attention_mask is None:
                encoder_attention_mask = torch.ones_like(
                    language_inputs["input_ids"],
                    dtype=torch.bool,
                )
            else:
                encoder_attention_mask = attention_mask.to(dtype=torch.bool)
        ctx.action_side["encoder_attention_mask_tensor"] = encoder_attention_mask.contiguous()

        print("compiling MolmoAct2 action flow step")
        ctx.diffusion_spec = self.hooks.build_diffusion_spec(ctx)
        action_dir = ctx.engine_subdir("action")
        ctx.handles["action"] = sink.export_diffusion(
            ctx.diffusion_spec,
            engine_dir=action_dir,
        )

        if accuracy_check:
            self.hooks.after_export(ctx, sink)

        ctx.plugin_info = self.hooks.finalize_plugin_info(ctx)
        return PipelineResult(handles=ctx.handles, plugin_info=ctx.plugin_info)
