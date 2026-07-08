from __future__ import annotations

import sys
import logging
import torch_tensorrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import copy

from pathlib import Path

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from lerobot.policies.molmoact2 import MolmoAct2Config, MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.policies.factory import make_pre_post_processors

from trt.modules.export.vision import GridVisionExportModule, TokenPoolingExportModule
from trt.modules.export.language import (
    MolmoTextEncoderKVExportModule,
    MolmoTextCausalLMExportModule,
    gather_last_token_hidden,
)
from trt.modules.export.diffusion import (
    GrootDiTStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
    TRTDynamicCategorySpecificMLPExportModule,
)
from trt.plugin.plugin_utils import (
    restore_attention,
    patch_molmo_language_attention,
    patch_molmo_vision_attention,
)
from trt.measure import parity
from trt.executor.models.groot.helpers import make_embodiment_id
from trt.data import create_pil_messages, prepare_model_inputs
from trt.utils import force_hf_attention
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.vision import nchw_to_hwc
from trt.rope import make_rope_rotary_cos_sin
from trt.data import (
    load_test_data, 
    frame_from_test_data,
    pack_state
)

from trt.plugin.plugin_utils import patch_vision_attention, patch_language_attention, patch_vision_attention_reference
from trt.compile import make_input_spec

from typing import Any, Sequence

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

LANGUAGE_TRT_SETTINGS = {
    **TRT_SETTINGS,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
}

def load_config(device):
    config = MolmoAct2Config(
        checkpoint_path="allenai/MolmoAct2",
        device=str(device),
        inference_action_mode="continuous",
        # Optional: loads LIBERO norm stats + prompt/camera metadata from the checkpoint
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.wrist_image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )
    config.validate_features()
    policy = MolmoAct2Policy(config).to(device).eval()
    return config, policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()
    
    dtype = torch.float16

    config, policy = load_config(device)
    model = policy.model.to(device=device, dtype=dtype).eval()
    backbone = model.model
    vision = backbone.vision_backbone
    language = backbone.transformer
    action_expert = backbone.action_expert

    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")

    #if action_expert is not None:
    #    force_hf_attention(action_expert, "eager")

    pre_processor, post_processor = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    pack_step = next(
        s for s in pre_processor.steps
        if isinstance(s, MolmoAct2PackInputsProcessorStep)
    )
    molmo_processor = pack_step.processor
    text_tok = molmo_processor.tokenizer

    data = load_test_data(
        "lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    frame = frame_from_test_data(data, policy, fill_missing=True)
    model_inputs = pre_processor(frame)

    input_ids = model_inputs["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = model_inputs["attention_mask"].to(device=device, dtype=torch.long)
    pixel_values = model_inputs["pixel_values"].to(device=device, dtype=dtype)
    image_token_pooling = model_inputs["image_token_pooling"].to(device=device)
    image_grids = model_inputs["image_grids"].to(device=device)
    image_num_crops = model_inputs["image_num_crops"].to(device=device)
    
    print('Compiling vision')

    # MolmoAct2 vision is a token-pooling backbone, not Eagle grid-vision.
    # merge_visual_inputs turns the flat processor tensors into the batched
    # (media, pooling_indices) the backbone consumes; the backbone returns one
    # row per valid prompt image token -> [num_valid_tokens, H] (already flat).
    media, pooling_indices = backbone.merge_visual_inputs(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_token_pooling=image_token_pooling,
        image_grids=image_grids,
        image_num_crops=image_num_crops,
    )
    media = media.to(device=device, dtype=dtype)
    pooling_indices = pooling_indices.to(device=device)

    visual = TokenPoolingExportModule(
        encoder=vision,
        sample_media=media,
        sample_pooling_indices=pooling_indices,
    ).eval().to(device=device, dtype=dtype)

    # --- Rung A: eager (UNPATCHED) ---
    with torch.no_grad():
        embs_eager = visual(media, pooling_indices)

    # --- Rung C: TRT compiled with Molmo image_vit self-attention plugin ---
    # image_pooling_2d is cross-attention with an attention mask, so keep it native
    # until there is a dedicated cross-attention plugin path.
    vision_batch = int(media.shape[0] * media.shape[1])
    vision_seq_len = int(media.shape[2])
    patched_vision = patch_molmo_vision_attention(
        vision,
        batch_size=vision_batch,
        seq_len=vision_seq_len,
    )
    try:
        exported = torch.export.export(
            visual, args=(media, pooling_indices), strict=False
        )
        input_specs = make_input_spec((media, pooling_indices))
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            embs_trt = trt_engine(media, pooling_indices)
    finally:
        restore_attention(patched_vision)

    parity("MolmoAct2 vision A vs C", embs_eager, embs_trt)

    # STEP 2a: backbone / language (encoder K/V prefill — not CausalLM splice)
    print("Compiling language")

    image_token_positions = (
        input_ids.reshape(-1) == int(backbone.config.image_patch_id)
    ).nonzero(as_tuple=False).flatten()

    language = backbone.transformer
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(language.blocks)

    # Build multimodal inputs_embeds for eager reference
    safe_input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
    inputs_embeds = language.wte(safe_input_ids)
    flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).clone()
    flat_embeds[image_token_positions] = (
        flat_embeds[image_token_positions] + embs_trt[: image_token_positions.numel()]
    )
    inputs_embeds = flat_embeds.reshape_as(inputs_embeds).to(
        device=device, dtype=dtype
    ).contiguous()

    bsz, seq_len, _ = inputs_embeds.shape
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(bsz, -1)

    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        device,
        language_model=language,
        position_ids=position_ids,
    )
    ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    kv_caches = [
        torch.zeros(
            bsz, 2, num_key_value_heads, seq_len, head_dim,
            device=device, dtype=dtype,
        )
        for _ in range(num_layers)
    ]

    flat_tensors = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        *kv_caches,
    )

    attention_bias = backbone._build_native_attention_bias(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        token_type_ids=None,
        past_key_values=None,
    )

    # --- Rung A: true eager (UNPATCHED) ---
    with torch.no_grad():
        eager_out = language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_bias,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            collect_layer_kv_states=True,
        )
    hidden_eager = eager_out.last_hidden_state
    encoder_kv_eager = [
        (
            backbone._cache_to_sequence(key),
            backbone._cache_to_sequence(value),
        )
        for key, value in eager_out.past_key_values
    ]
    encoder_k_eager = torch.stack([k for k, _ in encoder_kv_eager], dim=0).contiguous()
    encoder_v_eager = torch.stack([v for _, v in encoder_kv_eager], dim=0).contiguous()

    language_module = MolmoTextEncoderKVExportModule(
        language,
    ).eval().to(device=device)

    # --- Rung C: plugin + TRT compiled ---
    patched = patch_molmo_language_attention(
        language,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        enable_bidirectional_prefill=0,
    )
    try:
        with torch.no_grad():
            language_module(*flat_tensors)

        language_exported = torch.export.export(
            language_module, args=flat_tensors, strict=False
        )
        language_input_specs = make_input_spec(flat_tensors)
        language_trt = torch_tensorrt.dynamo.compile(
            language_exported,
            inputs=language_input_specs,
            **{**LANGUAGE_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            hidden_trt, encoder_k_trt, encoder_v_trt = language_trt(*flat_tensors)
    finally:
        restore_attention(patched)

    parity("MolmoAct2 language A vs C (TRT)", hidden_eager, hidden_trt)
    parity("MolmoAct2 encoder_k A vs C (TRT)", encoder_k_eager, encoder_k_trt)
    parity("MolmoAct2 encoder_v A vs C (TRT)", encoder_v_eager, encoder_v_trt)

    # STEP 2b: discrete language prefill (logits + prefix K/V — separate from action-expert K/V)
    print("Compiling discrete language")

    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.long)
    discrete_flat_tensors = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    with torch.no_grad():
        logits_eager = model.lm_head(
            gather_last_token_hidden(hidden_eager, last_token_ids)
        ).float()

    discrete_module = MolmoTextCausalLMExportModule(
        language,
        model.lm_head,
    ).eval().to(device=device)

    patched_discrete = patch_molmo_language_attention(
        language,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        enable_bidirectional_prefill=0,
        name="molmo-discrete-language",
    )
    try:
        with torch.no_grad():
            discrete_module(*discrete_flat_tensors)

        discrete_exported = torch.export.export(
            discrete_module, args=discrete_flat_tensors, strict=False
        )
        discrete_input_specs = make_input_spec(discrete_flat_tensors)
        discrete_trt = torch_tensorrt.dynamo.compile(
            discrete_exported,
            inputs=discrete_input_specs,
            **{**LANGUAGE_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            logits_trt, hidden_discrete_trt, prefix_k_trt, prefix_v_trt = discrete_trt(
                *discrete_flat_tensors
            )
    finally:
        restore_attention(patched_discrete)

    parity("MolmoAct2 discrete logits A vs C (TRT)", logits_eager, logits_trt)
    parity("MolmoAct2 discrete hidden A vs C (TRT)", hidden_eager, hidden_discrete_trt)

    print("COmpiling diffusion")

    return 0

if __name__ == "__main__":
    SystemExit(main())