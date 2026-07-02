"""Per-model helpers for vision ``plan_export`` hooks (pi05, smolvla)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trt.io_spec import PI05_EDGE_IO, PipelineIOSpec
from trt.plugin_utils import infer_smolvlm_seq_len
from trt.utils import clone_hf_module_for_export
from trt.vision import DEFAULT_VISION_TRT_SETTINGS


def build_pi05_vision_export_params(
    core: nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device,
    *,
    io: PipelineIOSpec = PI05_EDGE_IO,
    trt_settings: dict | None = None,
) -> dict[str, Any]:
    paligemma = core.paligemma_with_expert.paligemma
    pixel_values_nchw = pixel_values.to(device=device).contiguous()

    vision_tower = clone_hf_module_for_export(
        paligemma.model.vision_tower,
        device,
        dtype=next(paligemma.model.vision_tower.parameters()).dtype,
    )
    projector = clone_hf_module_for_export(
        paligemma.model.multi_modal_projector,
        device,
        dtype=next(paligemma.model.multi_modal_projector.parameters()).dtype,
        config=paligemma.config,
    )
    patch_vision_model = vision_tower.vision_model
    with torch.no_grad():
        siglip_hidden = patch_vision_model.embeddings(pixel_values=pixel_values_nchw)
    patch_batch_size = int(siglip_hidden.shape[0])
    patch_seq_len = int(siglip_hidden.shape[1])

    return {
        "visual_vision_model": vision_tower,
        "patch_vision_model": patch_vision_model,
        "projector": projector,
        "input_dtype": pixel_values_nchw.dtype,
        "patch_batch_size": patch_batch_size,
        "patch_seq_len": patch_seq_len,
        "config_seq_len": patch_seq_len,
        "vocab_size": int(paligemma.config.text_config.vocab_size),
        "image_token_id": int(getattr(paligemma.config, "image_token_index", 257152)),
        "force_float32_input": True,
        "cast_output_to_input_dtype": True,
        "io": io.vision,
        "trt_settings": dict(trt_settings or DEFAULT_VISION_TRT_SETTINGS),
    }


def build_smolvla_vision_export_params(
    core: nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device,
    *,
    io: PipelineIOSpec = PI05_EDGE_IO,
    trt_settings: dict | None = None,
) -> dict[str, Any]:
    vlm = core.vlm_with_expert.get_vlm_model()
    vision_dtype = next(vlm.vision_model.parameters()).dtype
    pixel_values_nchw = pixel_values.to(device=device, dtype=vision_dtype).contiguous()

    vision_model = clone_hf_module_for_export(
        vlm.vision_model,
        device,
        dtype=vision_dtype,
    )
    connector = clone_hf_module_for_export(
        vlm.connector,
        device,
        dtype=next(vlm.connector.parameters()).dtype,
    )
    patch_batch_size, patch_seq_len = infer_smolvlm_seq_len(vision_model, pixel_values_nchw)

    image_token_id = core.fake_image_token
    if hasattr(image_token_id, "item"):
        image_token_id = int(image_token_id.item())
    else:
        image_token_id = int(image_token_id)

    return {
        "visual_vision_model": vision_model,
        "patch_vision_model": vision_model,
        "projector": connector,
        "input_dtype": vision_dtype,
        "patch_batch_size": patch_batch_size,
        "patch_seq_len": patch_seq_len,
        "config_seq_len": 0,
        "vocab_size": int(vlm.config.text_config.vocab_size),
        "image_token_id": image_token_id,
        "patch_name": "SmolVLM",
        "allow_attention_mask": True,
        "io": io.vision,
        "trt_settings": dict(trt_settings or DEFAULT_VISION_TRT_SETTINGS),
    }
