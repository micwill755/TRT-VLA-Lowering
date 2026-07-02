from __future__ import annotations

"""GR00T vision export hooks: Eagle SigLIP → TRT VitRunner engine.

Stage 0 in the GROOT export pipeline. ``plan_export`` builds an ``ExportPlan``;
``compile`` traces ``GridVisionExportModule`` and writes ``visual/visual.engine``;
``metadata`` records dims for downstream stages and load/benchmark.
"""

from pathlib import Path

import torch

from trt.compile import save_trt_engine_module
from trt.hooks.export.plan import ExportPlan
from trt.io_spec import GROOT_EDGE_IO
from trt.modules.export.vision import GridVisionExportModule
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.runner.base import StageContext
from trt.utils import clone_hf_module_for_export
from trt.vision import (
    DEFAULT_VISION_TRT_SETTINGS as VISION_TRT_SETTINGS,
    nchw_to_hwc,
    vit_visual_edge_config,
)


def _pixel_values(ctx, stage_inputs: dict) -> torch.Tensor:
    """Resolve Eagle ``pixel_values`` for trace sample inputs.

    Preprocess (``groot/export/preprocess.py``) normally hoists pixels into
    ``ctx.export_state["pixel_values"]`` as fp16 ``[B, 3, H, W]``. Fall back to
    raw ``stage_inputs`` or nested ``tokenized_data`` when export is invoked
    without the pipeline preprocess hook.
    """
    export_state = getattr(ctx, "export_state", {})
    if "pixel_values" in export_state:
        return export_state["pixel_values"]
    if "pixel_values" in stage_inputs:
        return stage_inputs["pixel_values"]
    tokenized = stage_inputs.get("tokenized_data") or export_state.get("tokenized")
    if tokenized is not None and "pixel_values" in tokenized:
        return tokenized["pixel_values"]
    raise KeyError("pixel_values not found in export_state or stage inputs")


def plan_export(ctx: StageContext, stage_inputs: dict) -> ExportPlan:
    """Build the GR00T vision TRT export plan (Eagle SigLIP + mlp1 projector).

    Shape flow (libero example, 2 cameras @ 224×224)::

        pixel_values          [B, 3, H, W]   NCHW from Eagle processor (B ≈ num images)
        pixel_values_nchw     [B, 3, H, W]   fp16 on export device
        siglip_hidden         [B, S_vit, H_vit]  patch embeddings (probe for S_vit)
        images_hwc            [B, H, W, 3]   TRT / VitRunner engine input layout
        module forward out    [B * S_out, H_lm]  flattened rows for C++ embedding lookup

    ``S_vit`` is the SigLIP patch grid length (e.g. 256 for 224² / 14² patches).
    ``S_out`` is the LM-side sequence length after optional pixel-shuffle downsampling
    and projection — written to ``visual/config.json`` as ``seq_len``.
    """
    # --- 1. Sample pixels from preprocess ---------------------------------
    # Eagle processor output; typically B = number of camera frames in the batch.
    pixel_values = _pixel_values(ctx, stage_inputs)  # [B, 3, H, W]
    input_dtype = torch.float16

    eagle = ctx.model.backbone.eagle_model
    pixel_values_nchw = pixel_values.to(device=ctx.device, dtype=input_dtype).contiguous()

    # --- 2. Clone export subgraph (avoid mutating the live policy weights) -
    # vision_model: full Eagle vision tower wrapper (runs SigLIP + layer select).
    # projector:    eagle.mlp1 — maps vit hidden size → LM hidden size (1536).
    vision_model = clone_hf_module_for_export(
        eagle.vision_model,
        ctx.device,
        dtype=input_dtype,
    )
    projector = clone_hf_module_for_export(
        eagle.mlp1,
        ctx.device,
        dtype=input_dtype,
    )

    # Inner SigLIP module — target for attention patching during TRT compile.
    patch_vision_model = vision_model.vision_model

    # --- 3. Probe static patch grid from a dry-run embedding --------------
    # embeddings() is cheaper than a full forward pass but exposes [B, S_vit, H_vit]
    # so we know batch/seq dims for the TRT attention plugin and VitRunner config.
    with torch.no_grad():
        siglip_hidden = patch_vision_model.embeddings(pixel_values=pixel_values_nchw)
    patch_batch_size = int(siglip_hidden.shape[0])   # B
    patch_seq_len = int(siglip_hidden.shape[1])      # S_vit (e.g. 256)

    # --- 4. C++ VitRunner metadata (not part of the TRT graph) ------------
    # vocab_size / image_token_id: tell llm_inference where to splice vision rows
    # into the LM embedding table for <|image|> placeholders in the chat template.
    image_token_id = int(getattr(eagle, "image_token_index", eagle.config.image_token_index))
    vocab_size = int(eagle.language_model.config.vocab_size)

    # --- 5. GR00T-specific vision head options ----------------------------
    # select_layer: which hidden state to read (-1 = last_hidden_state).
    # pixel_shuffle + downsample_ratio: optional spatial downsampling before mlp1;
    # when enabled, S_out < S_vit (see GridVisionExportModule._apply_pixel_shuffle).
    select_layer = int(eagle.select_layer)
    pixel_shuffle = bool(eagle.use_pixel_shuffle)
    downsample_ratio = float(eagle.downsample_ratio)

    # --- 6. Layout conversion for TRT trace -------------------------------
    # Policy/processor use NCHW; compiled VitRunner engine binds HWC pixel_values.
    images_hwc = nchw_to_hwc(pixel_values_nchw)  # [B, H, W, 3]

    # --- 7. Trace target module -------------------------------------------
    # Wraps vision → (optional pixel shuffle) → mlp1 → flatten [B*S_out, H_lm].
    # Constructor dry-runs forward once to pin output_seq_len / output_hidden_size.
    module = GridVisionExportModule(
        vision_model=vision_model,
        projector=projector,
        sample_pixel_values=images_hwc,
        select_layer=select_layer,
        pixel_shuffle=pixel_shuffle,
        downsample_ratio=downsample_ratio,
        force_float32_input=True,       # trace in fp32; cast output back to input dtype
        cast_output_to_input_dtype=True,
    ).eval().to(ctx.device)

    # Tokens per image after projection — consumed by vit_visual_edge_config below.
    config_seq_len = int(patch_seq_len or module.output_seq_len)  # S_out

    # --- 8. ExportPlan — consumed by ExportRunner + compile() hook --------
    return ExportPlan(
        module=module,
        sample_inputs=(images_hwc,),  # trace args: ([B, H, W, 3],)
        input_names=tuple(GROOT_EDGE_IO.vision.input_names),    # ("pixel_values",)
        output_names=tuple(GROOT_EDGE_IO.vision.output_names),    # ("visual_embeds",)
        engine_dir=ctx.engine_root / "visual",
        engine_file="visual.engine",
        model_type="visual",
        component="vision",
        trt_settings=dict(VISION_TRT_SETTINGS),
        cleanup_modules=(module, vision_model),
        args={
            # compile() patches SigLIP attention to fixed [B, S_vit] before torch→TRT
            "patch_target": patch_vision_model,
            "patch_batch_size": patch_batch_size,
            "patch_seq_len": patch_seq_len,
            "patch_name": "",
            "allow_attention_mask": False,
            # merged into visual/config.json for C++ VitRunner at inference time
            "vocab_size": vocab_size,
            "image_token_id": image_token_id,
            "config_seq_len": config_seq_len,
            # downstream stage glue: export tensor name → inference state key
            "tensor_aliases": {"visual_embeds": "image_embs"},
        },
    )


def compile(plan: ExportPlan) -> Path:
    """Trace ``plan.module`` and write ``visual/visual.engine`` + ``config.json``.

    Runs on the **cloned** subgraph from ``plan_export`` (see
    ``clone_hf_module_for_export``). Steps:

    1. Patch SigLIP encoder attention → ``ViTPluginAttention`` with fixed
       ``[B, S_vit]`` shapes so torch→TRT emits the custom vit attention op.
    2. ``save_trt_engine_module`` — torch.export trace of
       ``GridVisionExportModule([B,H,W,3]) → [B*S_out, H_lm]``, then TensorRT
       compile. ``offload_module_to_cpu`` in ``trt_settings`` may move the clone
       to CPU during the build.
    3. ``vit_visual_edge_config`` is merged into ``visual/config.json`` for C++
       VitRunner (``vocab_size``, ``image_token_id``, ``seq_len``) — not part of
       the TRT graph.
    4. ``restore_attention`` in ``finally`` — undo patch on the clone before
       ``ExportRunner`` deletes ``plan.cleanup_modules``.

    Returns the path to ``visual.engine``. Inference loads this file only.
    """
    args = plan.args

    # Operates on patch_target (inner SigLIP inside the cloned vision tower).
    patched = patch_vision_attention(
        args["patch_target"],
        batch_size=args["patch_batch_size"],
        seq_len=args["patch_seq_len"],
        name=args["patch_name"],
        allow_attention_mask=args["allow_attention_mask"],
    )
    try:
        return save_trt_engine_module(
            plan.module,
            plan.sample_inputs,  # ([B, H, W, 3],) HWC
            plan.engine_dir,
            engine_file=plan.engine_file,
            model_type=plan.model_type or "visual",
            component=plan.component or "vision",
            input_names=list(plan.input_names),    # ["pixel_values"]
            output_names=list(plan.output_names),  # ["visual_embeds"]
            example_output=None,
            extra_config=vit_visual_edge_config(
                vocab_size=args["vocab_size"],
                image_token_id=args["image_token_id"],
                seq_len=args["config_seq_len"],  # S_out tokens per image
            ),
            trt_settings=plan.trt_settings,
        )
    finally:
        restore_attention(patched)


def metadata(ctx: StageContext, plan: ExportPlan) -> dict:
    """Return stage-0 artifacts stored on ``StageResult.metadata``.

    Saved alongside the engine path in ``ctx.artifacts["stage_0"]``. Language
    export and load/benchmark can read ``config_seq_len`` / ``output_hidden_size``
    without re-probing the cloned module after cleanup.
    """
    del ctx
    args = plan.args
    return {
        "config_seq_len": args["config_seq_len"],      # S_out for VitRunner
        "image_token_id": args["image_token_id"],      # chat-template splice id
        "output_hidden_size": int(plan.module.output_hidden_size),  # H_lm
    }
