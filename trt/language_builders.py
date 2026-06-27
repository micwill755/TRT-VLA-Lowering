"""Per-model builders for ``LanguageEngineSpec``."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from trt.io_spec import GROOT_EDGE_IO, PI05_EDGE_IO, PipelineIOSpec
from trt.language import (
    DEFAULT_LANGUAGE_TRT_SETTINGS,
    LanguageEngineSpec,
    compute_vit_expanded_seq_len,
    language_edge_trt_settings,
    language_head_dim,
    make_dummy_inputs_embeds,
)
from trt.utils import clone_hf_module_for_export


def build_pi05_language_export_params(
    core: nn.Module,
    prefix: dict,
    device: torch.device,
    *,
    io: PipelineIOSpec = PI05_EDGE_IO,
    trt_settings: dict | None = None,
    dtype: torch.dtype = torch.float16,
) -> LanguageEngineSpec:
    prefix_embs = prefix["inputs_embeds"].to(device=device, dtype=dtype).contiguous()
    batch_size = int(prefix_embs.shape[0])
    max_seq_len = int(prefix_embs.shape[1])

    lm = clone_hf_module_for_export(
        core.paligemma_with_expert.paligemma.model.language_model,
        device,
        dtype=dtype,
    )
    decoder = getattr(lm, "model", lm)
    cfg = lm.config
    paligemma_cfg = core.paligemma_with_expert.paligemma.config
    image_token_id = int(getattr(paligemma_cfg, "image_token_index", 257152))

    lm_head = clone_hf_module_for_export(
        core.paligemma_with_expert.paligemma.lm_head,
        device,
        dtype=dtype,
    )

    return LanguageEngineSpec(
        decoder=decoder,
        lm_head=lm_head,
        language_model=lm,
        prefix_embs=prefix_embs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hidden_size=int(cfg.hidden_size),
        num_layers=int(cfg.num_hidden_layers),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        image_token_id=image_token_id,
        position_ids=prefix.get("position_ids"),
        enable_bidirectional_prefill=1,
        static_prefill_seq_len=False,
        export_dtype=dtype,
        io=io.language,
        trt_settings=dict(trt_settings or DEFAULT_LANGUAGE_TRT_SETTINGS),
        model_type="language",
        log_prefix="pi05",
    )


def build_smolvla_language_export_params(
    core: nn.Module,
    prefix: dict,
    device: torch.device,
    *,
    io: PipelineIOSpec = PI05_EDGE_IO,
    trt_settings: dict | None = None,
    dtype: torch.dtype = torch.float16,
) -> LanguageEngineSpec:
    prefix_embs = prefix["inputs_embeds"].to(device=device, dtype=dtype).contiguous()
    batch_size = int(prefix_embs.shape[0])
    max_seq_len = int(prefix_embs.shape[1])

    text_model = core.vlm_with_expert.get_vlm_model().text_model
    num_layers = int(core.vlm_with_expert.num_vlm_layers)
    lm_config = copy.deepcopy(text_model.config)
    lm_config.num_hidden_layers = num_layers
    lm = clone_hf_module_for_export(
        text_model,
        device,
        dtype=dtype,
        config=lm_config,
    )
    decoder = getattr(lm, "model", lm)
    cfg = lm.config

    image_token_id = core.fake_image_token
    if hasattr(image_token_id, "item"):
        image_token_id = int(image_token_id.item())
    else:
        image_token_id = int(image_token_id)

    lm_head = clone_hf_module_for_export(
        core.vlm_with_expert.vlm.lm_head,
        device,
        dtype=dtype,
    )

    return LanguageEngineSpec(
        decoder=decoder,
        lm_head=lm_head,
        language_model=lm,
        prefix_embs=prefix_embs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hidden_size=int(cfg.hidden_size),
        num_layers=num_layers,
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        image_token_id=image_token_id,
        position_ids=prefix.get("position_ids"),
        enable_bidirectional_prefill=1,
        static_prefill_seq_len=True,
        export_dtype=dtype,
        io=io.language,
        trt_settings=dict(trt_settings or DEFAULT_LANGUAGE_TRT_SETTINGS),
        model_type="smolvla",
        log_prefix="smolvla",
    )


def build_groot_language_export_params(
    core: nn.Module,
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    seq_len_per_image: int,
    device: torch.device,
    io: PipelineIOSpec = GROOT_EDGE_IO,
    trt_settings: dict | None = None,
    dtype: torch.dtype = torch.float16,
) -> LanguageEngineSpec:
    source_lm = core.backbone.eagle_model.language_model
    source_decoder = getattr(source_lm, "model", source_lm)
    num_layers = len(source_decoder.layers)
    lm_config = copy.deepcopy(source_lm.config)
    lm_config.num_hidden_layers = num_layers

    language_model = clone_hf_module_for_export(
        source_lm,
        device,
        dtype=dtype,
        config=lm_config,
    )
    decoder = getattr(language_model, "model", language_model)
    cfg = language_model.config
    head_dim = language_head_dim(cfg)

    max_seq_len = compute_vit_expanded_seq_len(
        input_ids,
        int(image_token_id),
        int(seq_len_per_image),
    )
    batch_size = int(input_ids.shape[0])
    prefix_embs = make_dummy_inputs_embeds(
        batch_size,
        max_seq_len,
        int(cfg.hidden_size),
        device=device,
        dtype=dtype,
    )

    lm_head = clone_hf_module_for_export(language_model.lm_head, device, dtype=dtype)

    return LanguageEngineSpec(
        decoder=decoder,
        lm_head=lm_head,
        language_model=language_model,
        prefix_embs=prefix_embs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hidden_size=int(cfg.hidden_size),
        num_layers=num_layers,
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=head_dim,
        image_token_id=int(image_token_id),
        seq_len_per_image=int(seq_len_per_image),
        select_layer=-1,
        enable_bidirectional_prefill=0,
        static_prefill_seq_len=False,
        export_dtype=dtype,
        io=io.language,
        trt_settings=dict(trt_settings or language_edge_trt_settings()),
        model_type="language",
        log_prefix="groot",
    )
