from __future__ import annotations

import sys
import logging
import torch_tensorrt
import torch
import argparse
import logging
import copy

from pathlib import Path

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from trt.plugin.plugin_utils import restore_attention
from trt.data import load_test_data, frame_from_test_data
from trt.modules.export.vision import GridVisionExportModule
from trt.modules.export.language import CausalLMExportModule, ContextProjectionExportModule
from trt.modules.export.diffusion import (
    PI05PrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
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

from typing import Any

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
    #"offload_module_to_cpu": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
}

def build_pi05_prefix_embs(
    pi05_model,
    img_masks,
    tokens,
    masks,
    image_embs_list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []

    for img_emb, img_mask in zip(image_embs_list, img_masks, strict=True):
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

    lang_emb = pi05_model.paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)
    pad_masks.append(masks)

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    valid = prefix_pad_masks.to(device=prefix_embs.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "build_pi05_prefix_embs requires equal valid token counts across the batch"
        )

    compact_len = int(valid_counts[0].item())
    compact_embs = torch.stack(
        [prefix_embs[b, valid[b], :] for b in range(prefix_embs.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [prefix_position_ids[b, valid[b]] for b in range(prefix_position_ids.shape[0])],
        dim=0,
    )
    compact_pad_mask = torch.ones(
        prefix_embs.shape[0],
        compact_len,
        device=prefix_pad_masks.device,
        dtype=torch.bool,
    )
    compact_attention_mask = torch.zeros(
        prefix_embs.shape[0],
        1,
        compact_len,
        compact_len,
        device=prefix_embs.device,
        dtype=torch.float32,
    )
    return compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids

def make_pi05_suffix_position_and_mask(core, prefix_pad_masks, x_t, device):
    batch_size, suffix_len = x_t.shape[:2]
    prefix_pad_masks = prefix_pad_masks.to(device=device)
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + [0] * (suffix_len - 1),
        dtype=torch.int64,
        device=device,
    )[None, :].expand(batch_size, -1)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    attention_mask = core._prepare_attention_masks_4d(full_att_2d_masks)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask

def load_config(device):
    config = PI05Config(
        device=str(device),
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=32,      # PI05 default is 32, not 64
        max_action_dim=32,
        image_resolution=(224, 224),
        input_features={
            f"{OBS_IMAGES}.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            f"{OBS_IMAGES}.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(
                type=FeatureType.STATE, shape=(32,)  # padded to max_state_dim
            ),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    )
    config.validate_features()
    policy = PI05Policy(config).to(device).eval()
    return config, policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()
    
    dtype = torch.float16

    # step 1 - load policy, retrieve vision, lm diffusion, 
    # create processors, data sample and replace attention
    config, policy = load_config(device)
    model = policy.model.to(device=device, dtype=dtype).eval()
    paligemma = model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    language = paligemma.language_model
    select_layer = -1

    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")

    pre_processor, post_processor = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    data = load_test_data(
        "lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    frame = frame_from_test_data(data, policy, fill_missing=True)
    model_inputs = pre_processor(frame)

    # PI05 batch prep (not GR00T tokenized_data)
    images, img_masks = policy._preprocess_images(model_inputs)
    tokens = model_inputs[OBS_LANGUAGE_TOKENS].to(device=device, dtype=torch.long)
    masks = model_inputs[OBS_LANGUAGE_ATTENTION_MASK].to(device=device, dtype=torch.bool)
    
    # step 2 vision
    pixel_values = torch.cat(
        [img.to(device=device, dtype=dtype) for img in images],
        dim=0,
    ).contiguous()
    projector = paligemma.multi_modal_projector

    # step 2: vision
    visual = GridVisionExportModule(
        vision_model=vision,                 # paligemma.vision_tower
        projector=projector,                 # paligemma.multi_modal_projector
        sample_pixel_values=pixel_values,
        select_layer=-1,                     # PI05 has no eagle.select_layer
        pixel_shuffle=False,
        downsample_ratio=0.5,
        force_float32_input=True,            # PI05 vision tower runs fp32 internally
        vision_kwargs={},
    ).eval().to(device=device, dtype=dtype)

    # --- Rung A: eager SDPA (UNPATCHED) ---
    with torch.no_grad():
        embs_eager = visual(pixel_values)

    # --- Patch SigLIP attention -> ViTPluginAttention ---
    hidden_states = vision.vision_model.embeddings(pixel_values.float())
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    patched = patch_vision_attention(
        vision.vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    try:
        # --- Rung B: eager with plugin attention (usually invalid eagerly) ---
        with torch.no_grad():
            embs_eager_plugin = visual(pixel_values)

        # --- Rung C: TRT compiled from patched module ---
        exported = torch.export.export(visual, args=(pixel_values,), strict=False)
        input_specs = make_input_spec((pixel_values,))
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            embs_trt = trt_engine(pixel_values)
            
    finally:
        restore_attention(patched)

    parity("PI05 vision A vs C", embs_eager, embs_trt)

    # step 3 language
    print("Compiling language")
    lm_head = model.paligemma_with_expert.paligemma.lm_head
    decoder = getattr(language, "model", language)

    per_camera_batch = int(images[0].shape[0])
    trt_image_embs = list(
        embs_trt.reshape(len(images), per_camera_batch, -1, embs_trt.shape[-1])
    )
    inputs_embeds, prefix_pad_mask, prefix_attention_mask, prefix_position_ids = build_pi05_prefix_embs(
        model,
        img_masks,
        tokens,
        masks,
        trt_image_embs,
    )

    bsz, seq_len, hidden = inputs_embeds.shape
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    lm_dtype = next(language.parameters()).dtype

    with torch.no_grad():
        eager_out = language(
            inputs_embeds=inputs_embeds.to(dtype=lm_dtype),
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    lm_hidden_eager = eager_out.last_hidden_state

    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(decoder.layers)

    lm = CausalLMExportModule(
        decoder,
        lm_head,
        select_layer=-1,
    ).eval().to(device=device)

    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        device,
        language_model=language,
        position_ids=prefix_position_ids,
    )
    ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.int64)
    kv_caches = [
        torch.zeros(
            bsz,
            2,
            num_key_value_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    flat_tensors = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    # PI05 prefix attends bidirectionally; patch_language_attention wires this
    # flag into the plugin config that the TRT converter reads at compile time.
    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        enable_bidirectional_prefill=1,
    )
    try:
        with torch.no_grad():
            _, lm_hidden_trt_ref, _, _ = lm(*flat_tensors)

        lm_exported = torch.export.export(lm, args=flat_tensors, strict=False)
        lm_input_specs = make_input_spec(flat_tensors)
        lm_trt_engine = torch_tensorrt.dynamo.compile(
            lm_exported,
            inputs=lm_input_specs,
            **{**LANGUAGE_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            trt_out = lm_trt_engine(*flat_tensors)
    finally:
        restore_attention(patched)

    parity("PI05 language A vs C (TRT)", lm_hidden_eager, trt_out[1])

    # step 4 diffusion (no action-context stage for PI05)
    print("Compiling diffusion")
    force_hf_attention(model.paligemma_with_expert.gemma_expert.model, "eager")

    prefix_k = trt_out[2].to(device=device, dtype=dtype).contiguous()
    prefix_v = trt_out[3].to(device=device, dtype=dtype).contiguous()

    diffusion_model = StaticActionVelocityStepExportModule(
        step_encoder=PI05PrefixKVStepEncoderExportModule(model),
        action_expert=model.paligemma_with_expert.gemma_expert.model,
        velocity_decoder=model.action_out_proj,
        output_tokens=model.config.chunk_size,
        cast_hidden_fp32=False,
    ).eval().to(device=device)

    step_actions = torch.randn(
        bsz,
        model.config.chunk_size,
        model.config.max_action_dim,
        device=device,
        dtype=dtype,
    )
    step_timestep = torch.full(
        (bsz,),
        1.0,
        device=device,
        dtype=torch.float32,
    )
    suffix_position_ids, suffix_attention_mask = make_pi05_suffix_position_and_mask(
        model,
        prefix_pad_mask,
        step_actions,
        device,
    )
    diffusion_input = (
        step_actions,
        step_timestep,
        prefix_k,
        prefix_v,
        suffix_position_ids,
        suffix_attention_mask,
    )

    with torch.no_grad():
        eager_velocity = diffusion_model(*diffusion_input)

    diffusion_exported = torch.export.export(diffusion_model, args=diffusion_input, strict=False)
    diffusion_input_specs = make_input_spec(diffusion_input)
    diffusion_trt_engine = torch_tensorrt.dynamo.compile(
        diffusion_exported,
        inputs=diffusion_input_specs,
        **{**ACTION_TRT_SETTINGS, "use_python_runtime": True},
    )
    with torch.no_grad():
        trt_velocity = diffusion_trt_engine(*diffusion_input)

    parity("PI05 diffusion A vs C (TRT)", eager_velocity, trt_velocity)

    return 0

if __name__ == "__main__":
    SystemExit(main())