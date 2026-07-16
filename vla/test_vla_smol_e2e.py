from __future__ import annotations

import sys
import logging
import math
import torch_tensorrt
import torch
import argparse
import logging
import copy
import time

from pathlib import Path

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.factory import make_pre_post_processors

from trt.modules.export.vision import GridVisionExportModule
from trt.modules.export.language import CausalLMExportModule
from trt.modules.export.diffusion import (
    SmolVLAPrefixKVStepEncoderExportModule,
    SmolVLAExpertExportModule,
    StaticActionVelocityStepExportModule,
)
from trt.plugin.plugin_utils import restore_attention
from trt.measure import parity
from trt.utils import force_hf_attention
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.rope import make_rope_rotary_cos_sin
from trt.data import (
    load_test_data, 
    frame_from_test_data,
)

from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import patch_vision_attention, patch_language_attention, patch_vision_attention_reference, infer_smolvlm_seq_len
from trt.compile import make_input_spec

from typing import Any

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
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

def build_smolvla_prefix_embs(
    smolvla_model,
    img_masks,
    lang_tokens,
    lang_masks,
    image_embs_list,
    state,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact SmolVLA prefix embeddings using TRT vision outputs."""
    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []

    for img_emb, img_mask in zip(image_embs_list, img_masks, strict=True):
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * (img_emb_dim**0.5)
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

    lang_emb = smolvla_model.vlm_with_expert.embed_language_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)
    embs.append(lang_emb)
    pad_masks.append(lang_masks)

    state_emb = smolvla_model.state_proj(state)
    state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
    embs.append(state_emb)
    bsize = state_emb.shape[0]
    device = state_emb.device
    states_seq_len = state_emb.shape[1]
    pad_masks.append(torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device))

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    valid = prefix_pad_masks.to(device=prefix_embs.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "build_smolvla_prefix_embs requires equal valid token counts across the batch"
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
        dtype=compact_embs.dtype,
    )
    return compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids

def make_smolvla_suffix_position_and_mask(model, prefix_pad_masks, x_t, timestep):
    """Build suffix position ids and full 2D attention mask for one denoise step."""
    _, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_masks = prefix_pad_masks.to(device=x_t.device)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, full_att_2d_masks

def load_config(device):
    config = SmolVLAConfig(
        device=str(device),
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=32,       # SmolVLA default (GR00T uses 64)
        max_action_dim=32,
        resize_imgs_with_padding=(512, 512),  # not image_size=(224, 224)
        load_vlm_weights=True,  # required for real VLM weights
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        input_features={
            f"{OBS_IMAGES}.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            f"{OBS_IMAGES}.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    )
    config.validate_features()
    policy = SmolVLAPolicy(config).to(device).eval()
    return config, policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()
    
    dtype = torch.float16

    config, policy = load_config(device)
    model = policy.model.to(device=device, dtype=dtype).eval()
    vlm_with_expert = model.vlm_with_expert
    vlm = vlm_with_expert.get_vlm_model()
    vision = vlm.vision_model
    language = vlm.text_model
    connector = vlm.connector
    select_layer = -1

    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")
    language.config._attn_implementation = "sdpa"
    language = language.to(device=device, dtype=dtype).eval()

    pre_processor, post_processor = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    text_tok = vlm_with_expert.processor.tokenizer

    data = load_test_data(
        "lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    frame = frame_from_test_data(data, policy, fill_missing=True)
    model_inputs = pre_processor(frame)

    images, img_masks = policy.prepare_images(model_inputs)
    tokens = model_inputs[OBS_LANGUAGE_TOKENS].to(device=device, dtype=torch.long)
    masks = model_inputs[OBS_LANGUAGE_ATTENTION_MASK].to(device=device, dtype=torch.bool)
    state = policy.prepare_state(model_inputs).to(device=device, dtype=dtype).contiguous()

    pixel_values = torch.cat(
        [img.to(device=device, dtype=dtype) for img in images],
        dim=0,
    ).contiguous()

    # Aliases for downstream compile steps.
    input_ids = tokens
    attention_mask = masks.to(dtype=torch.long)

    print('Compiling vision')

    # add vision wrapper — SmolVLM vision transformer + connector. The connector
    # already applies pixel shuffle + modality projection, so no extra shuffle here.
    visual = GridVisionExportModule(
        vision_model=vision,        # vlm.vision_model (SmolVLM vision transformer)
        projector=connector,        # vlm.connector
        sample_pixel_values=pixel_values,
        select_layer=-1,            # use last_hidden_state
        pixel_shuffle=False,        # connector already applies pixel shuffle
        vision_kwargs={},
    ).eval().to(device=device, dtype=dtype)

    # --- Rung A: eager SDPA (UNPATCHED) ---
    with torch.no_grad():
        embs_eager = visual(pixel_values)

    for _ in range(5):
        visual(pixel_values)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        visual(pixel_values)
    end.record()
    torch.cuda.synchronize()
    vision_eager_elapsed_ms = start.elapsed_time(end) / 100

    # --- Patch SigLIP attention -> ViTPluginAttention ---
    # SmolVLM vision transformer exposes .embeddings/.encoder directly (no nested
    # .vision_model like Eagle); embeddings needs a patch_attention_mask.
    batch_size, seq_len = infer_smolvlm_seq_len(vision, pixel_values)
    patched = patch_vision_attention(
        vision,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
        allow_attention_mask=True,
    )
    # replace using self implemented siglip attention to compare
    '''
    patched = patch_vision_attention_reference(
        vision
    )'''

    try:
        # --- Rung B: eager, but now with the plugin attention ---
        with torch.no_grad():
            embs_eager_plugin = visual(pixel_values)

        # --- Rung C: TRT compiled from the patched module ---
        exported = torch.export.export(visual, args=(pixel_values,), strict=False)
        input_specs = make_input_spec((pixel_values,))
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **VISION_TRT_SETTINGS,
        )
        with torch.no_grad():
            embs_trt = trt_engine(pixel_values)

        for _ in range(5):
            trt_engine(pixel_values)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            trt_engine(pixel_values)
        end.record()
        torch.cuda.synchronize()
        vision_trt_elapsed_ms = start.elapsed_time(end) / 100
    finally:
        restore_attention(patched)

    parity("A vs C (prod)", embs_eager, embs_trt)
    
    # STEP 2 language
    print('Compiling language')
    lm_head = vlm_with_expert.vlm.lm_head
    decoder = getattr(language, "model", language)

    per_camera_batch = int(images[0].shape[0])
    trt_image_embs = list(
        embs_trt.reshape(len(images), per_camera_batch, -1, embs_trt.shape[-1])
    )
    inputs_embeds, prefix_pad_mask, prefix_attention_mask, prefix_position_ids = (
        build_smolvla_prefix_embs(
            model,
            img_masks,
            tokens,
            masks,
            trt_image_embs,
            state,
        )
    )

    bsz, seq_len, hidden = inputs_embeds.shape
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    lm_dtype = next(language.parameters()).dtype
    prefix_attention_mask = prefix_attention_mask.to(device=device, dtype=lm_dtype)

    # ---------------------------------------------------------------------------
    # RUNG A: eager language — HF text_model on compact prefix embeddings
    # ---------------------------------------------------------------------------
    # warmup (compile/caches/autotune settle) — do a few before timing
    for _ in range(5):
        language(
            inputs_embeds=inputs_embeds.to(dtype=lm_dtype),
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        eager_out = language(
            inputs_embeds=inputs_embeds.to(dtype=lm_dtype),
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    end.record()
    torch.cuda.synchronize()
    eager_elapsed_ms = start.elapsed_time(end) / 100

    lm_hidden_eager = eager_out.last_hidden_state

    # ---------------------------------------------------------------------------
    # RUNG C: plugin + CausalLMExportModule — patch first, then flat ABI
    # ---------------------------------------------------------------------------
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
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)   # fresh prefill
    flat_tensors = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    # SmolVLA prefix (images + language + state) attends bidirectionally within
    # the compact valid sequence, same as PI0.5 prefix prefill.
    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        context_attention_mask_type=ContextAttentionMaskType.PADDING,
    )

    try:
        with torch.no_grad():
            logits, lm_hidden_trt_ref, prefix_k, prefix_v = lm(*flat_tensors)

        lm_exported = torch.export.export(lm, args=flat_tensors, strict=False)
        lm_input_specs = make_input_spec(flat_tensors)
        lm_trt_engine = torch_tensorrt.dynamo.compile(
            lm_exported,
            inputs=lm_input_specs,
            **LANGUAGE_TRT_SETTINGS,
        )

        for _ in range(5):
            with torch.no_grad():
                lm_trt_engine(*flat_tensors)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            trt_out = lm_trt_engine(*flat_tensors)
        end.record()
        torch.cuda.synchronize()
        #print(f"Avg latency: {start.elapsed_time(end) / 100:.3f} ms")
        trt_elapsed_ms = start.elapsed_time(end) / 100

    finally:
        restore_attention(patched)

    parity("language A vs C (TRT)", lm_hidden_eager, trt_out[1])

    # step 3 diffusion (no action-context stage for SmolVLA)
    print("compiling diffusion")

    prefix_k = trt_out[2].to(device=device, dtype=dtype).contiguous()
    prefix_v = trt_out[3].to(device=device, dtype=dtype).contiguous()

    diffusion_model = StaticActionVelocityStepExportModule(
        step_encoder=SmolVLAPrefixKVStepEncoderExportModule(model),
        action_expert=SmolVLAExpertExportModule(model.vlm_with_expert),
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
    suffix_position_ids, suffix_attention_mask = make_smolvla_suffix_position_and_mask(
        model,
        prefix_pad_mask,
        step_actions,
        step_timestep,
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

    for _ in range(5):
        diffusion_model(*diffusion_input)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        diffusion_model(*diffusion_input)
    end.record()
    torch.cuda.synchronize()
    diffusion_eager_elapsed_ms = start.elapsed_time(end) / 100

    diffusion_exported = torch.export.export(diffusion_model, args=diffusion_input, strict=False)
    diffusion_input_specs = make_input_spec(diffusion_input)
    diffusion_trt_engine = torch_tensorrt.dynamo.compile(
        diffusion_exported,
        inputs=diffusion_input_specs,
        **ACTION_TRT_SETTINGS,
    )
    with torch.no_grad():
        trt_velocity = diffusion_trt_engine(*diffusion_input)

    for _ in range(5):
        diffusion_trt_engine(*diffusion_input)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        diffusion_trt_engine(*diffusion_input)
    end.record()
    torch.cuda.synchronize()
    diffusion_trt_elapsed_ms = start.elapsed_time(end) / 100

    parity("diffusion step A vs C", eager_velocity, trt_velocity)

    eager_total_ms = vision_eager_elapsed_ms + eager_elapsed_ms + diffusion_eager_elapsed_ms
    trt_total_ms = vision_trt_elapsed_ms + trt_elapsed_ms + diffusion_trt_elapsed_ms

    print(f"vision eager execute: {vision_eager_elapsed_ms:.3f} ms")
    print(f"vision trt execute: {vision_trt_elapsed_ms:.3f} ms")
    print(f"vision speedup: {(vision_eager_elapsed_ms / vision_trt_elapsed_ms):.3f}x")
    print(f"lm eager execute: {eager_elapsed_ms:.3f} ms")
    print(f"lm trt execute: {trt_elapsed_ms:.3f} ms")
    print(f"lm speedup: {(eager_elapsed_ms / trt_elapsed_ms):.3f}x")
    print(f"diffusion eager execute: {diffusion_eager_elapsed_ms:.3f} ms")
    print(f"diffusion trt execute: {diffusion_trt_elapsed_ms:.3f} ms")
    print(f"diffusion speedup: {(diffusion_eager_elapsed_ms / diffusion_trt_elapsed_ms):.3f}x")
    print(f"total eager execute: {eager_total_ms:.3f} ms")
    print(f"total trt execute: {trt_total_ms:.3f} ms")
    print(f"total speedup: {(eager_total_ms / trt_total_ms):.3f}x")
    return 0

if __name__ == "__main__":
    SystemExit(main())