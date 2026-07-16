from __future__ import annotations

import sys
import logging
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

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep

from trt.modules.export.vision import GridVisionExportModule
from trt.modules.export.language import CausalLMExportModule, ContextProjectionExportModule
from trt.modules.export.diffusion import (
    GrootDiTStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
    TRTDynamicCategorySpecificMLPExportModule,
)
from trt.plugin.plugin_utils import restore_attention
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

from trt.plugin.attention import ContextAttentionMaskType
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

def load_config(device):
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.5-3B",
        device=str(device),
        embodiment_tag="new_embodiment",  # or "gr1", "oxe_droid", etc.
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=64,
        max_action_dim=32,
        image_size=(224, 224),
        tokenizer_assets_repo="lerobot/eagle2hg-processor-groot-n1p5",
        # Match lerobot/libero camera keys (see Test/trt/data.py IMAGE_KEYS)
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    )

    policy = GrootPolicy(config).to(device).eval()
    return config, policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()
    
    dtype = torch.float16

    config, policy = load_config(device)
    model = policy._groot_model
    eagle = model.backbone.eagle_model
    vision = model.backbone.eagle_model.vision_model
    language = model.backbone.eagle_model.language_model
    # wrapper from creates layers from 0-11 so -1 is the last hidden state for both eager and trt compiled
    select_layer = -1 #model.backbone.select_layer

    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")
    language.config._attn_implementation = "sdpa"
    language = language.to(device=device, dtype=dtype).eval()

    pre_processor, post_processor = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    # get tokenizer
    eagle_step = next(
        s for s in pre_processor.steps
        if isinstance(s, GrootEagleEncodeStep)
    )
    proc = eagle_step.proc
    eagle_processor = proc
    text_tok = getattr(proc, "tokenizer", proc)

    data = load_test_data(
        "lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    messages = create_pil_messages(data)
    text = eagle_processor.apply_chat_template(
        messages,
        tokenize=False,
        **{"add_generation_prompt": True},
    )

    image_inputs, video_inputs = eagle_processor.process_vision_info(messages)

    tokenized_data = eagle_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **{
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            }
        },
    )

    model_inputs = {
        "tokenized_data": tokenized_data,
        "state": data["state"],
        "task": data["task"],
    }
    
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = tokenized_data["attention_mask"].to(device=device, dtype=torch.long)
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=dtype)
    state = pack_state(
        model_inputs["state"],  # [7] libero proprio
        max_state_dim=64,  # 64
        device=device,
    ) 
    state = state.to(device=device, dtype=dtype).contiguous()
    embodiment_id = make_embodiment_id(policy, state, device, torch.long)
    
    print('Compiling vision')

    # add vision wrapper
    visual = GridVisionExportModule(
        vision_model=vision,
        projector=eagle.mlp1,
        sample_pixel_values=pixel_values,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
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
    hidden_states = vision.vision_model.embeddings(pixel_values)
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    patched = patch_vision_attention(
        vision.vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    # replace using self implemented siglip attention to compare
    '''
    patched = patch_vision_attention_reference(
        vision.vision_model
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
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
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
        # always undo the patch so later eager runs aren't affected
        restore_attention(patched)

    # --- Localize the error ---
    parity("A vs C (prod)", embs_eager, embs_trt)
    # two below are not valid, 
    # b attention outputs will be zeros - it will run empty custom_op operations
    # c is running kernels producing correct attention outputs
    #parity("A vs B (plugin)", embs_eager, embs_eager_plugin)
    #parity("B vs C (trt only)", embs_eager_plugin, embs_trt)

    # STEP 2 language
    print('Compiling language')
    image_token_index = getattr(
        eagle,
        "image_token_index",
        eagle.config.image_token_index,
    )

    # ---------------------------------------------------------------------------
    # Shared: build packed language embeddings (same for both paths)
    # ---------------------------------------------------------------------------
    input_embs = language.get_input_embeddings()(input_ids)
    bsz, seq_len, hidden = input_embs.shape

    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)
    image_token_mask = flat_ids == image_token_index

    # Eager language input: eager vision -> eager LM
    num_slots = int(image_token_mask.sum().item())
    flat_embs_eager = input_embs.clone().reshape(bsz * seq_len, hidden)
    flat_image_embs_eager = embs_eager.reshape(-1, hidden).to(
        device=flat_embs_eager.device,
        dtype=flat_embs_eager.dtype,
    )
    flat_embs_eager[image_token_mask] = flat_image_embs_eager[:num_slots]
    inputs_embeds_eager = flat_embs_eager.reshape(bsz, seq_len, hidden).to(
        device=device,
        dtype=dtype,
    ).contiguous()

    # TRT language input: TRT vision -> TRT LM
    flat_embs_trt = input_embs.clone().reshape(bsz * seq_len, hidden)
    flat_image_embs_trt = embs_trt.reshape(-1, hidden).to(
        device=flat_embs_trt.device,
        dtype=flat_embs_trt.dtype,
    )
    flat_embs_trt[image_token_mask] = flat_image_embs_trt[:num_slots]
    inputs_embeds_trt = flat_embs_trt.reshape(bsz, seq_len, hidden).to(
        device=device,
        dtype=dtype,
    ).contiguous()
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # ---------------------------------------------------------------------------
    # RUNG A: eager language — no wrapper, no patch
    # HF owns RoPE, KV cache, layer loop, norm
    # ---------------------------------------------------------------------------
    for _ in range(5):
        language(
            inputs_embeds=inputs_embeds_trt,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            inputs_embeds=inputs_embeds_trt,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    end.record()
    torch.cuda.synchronize()
    eager_elapsed_ms = start.elapsed_time(end) / 100

    lm_hidden_eager = eager_out.hidden_states[select_layer]

    # ---------------------------------------------------------------------------
    # RUNG C: plugin + CausalLMExportModule — patch first, then flat ABI
    # ---------------------------------------------------------------------------
    decoder = getattr(language, "model", language)   # Qwen3Model with .layers
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(decoder.layers)

    lm = CausalLMExportModule(
        decoder,
        language.lm_head,
        select_layer=select_layer,
    ).eval().to(device=device)

    #pad_mask = attention_mask.to(device=device, dtype=torch.bool)
    # build position_ids for the real sequence:
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        device,
        language_model=language,
        position_ids=position_ids,
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
        inputs_embeds_trt,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )

    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        context_attention_mask_type=ContextAttentionMaskType.CAUSAL,
    )

    try:
        with torch.no_grad():
            logits, lm_hidden_trt_ref, prefix_k, prefix_v = lm(*flat_tensors)
            # lm_hidden_trt_ref is what CausalLMExportModule calls context_hidden
            # compare lm_hidden_eager vs lm_hidden_trt_ref (after TRT compile, same flat_tensors in)
    
        lm_exported = torch.export.export(lm, args=flat_tensors, strict=False)
        lm_input_specs = make_input_spec(flat_tensors)
        lm_trt_engine = torch_tensorrt.dynamo.compile(
            lm_exported,
            inputs=lm_input_specs,
            **{**LANGUAGE_TRT_SETTINGS, "use_python_runtime": True},
        )

        with torch.no_grad():
            trt_out = lm_trt_engine(*flat_tensors)
            # trt_out is tuple: (logits, context_hidden, prefix_k, prefix_v)

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
        trt_elapsed_ms = start.elapsed_time(end) / 100

    finally:
        restore_attention(patched)

    parity("language A vs C (TRT)", lm_hidden_eager, trt_out[1])

    # step 3 action context
    print("compiling action context")

    # pre action transformation - action context
    eager_context_input = model.backbone.eagle_linear(lm_hidden_eager).to(dtype=dtype)
    trt_context_input = trt_out[1].to(dtype=dtype)

    # input hidden_states  [B, S, 2048]   ← language context_hidden (lm_hidden_eager or trt_out[1])
    action_context = ContextProjectionExportModule(
        model.backbone.eagle_linear, # [B, S, 1536],
        model.action_head.vlln, # [B, S, 1536] (LayerNorm),
        model.action_head.vl_self_attention # [B, S, 1536] (vl_self_attention (SelfAttentionTransformer)) -> [output],
    ).eval().to(dtype=dtype)

    with torch.no_grad():
        # context hidden as input
        eager_context_embs = action_context(eager_context_input)

    for _ in range(5):
        action_context(eager_context_input)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        action_context(eager_context_input)
    end.record()
    torch.cuda.synchronize()
    action_context_eager_elapsed_ms = start.elapsed_time(end) / 100

    action_context_exported = torch.export.export(action_context, args=(trt_context_input,), strict=False)
    action_context_input_specs = make_input_spec((trt_context_input,))
    action_context_trt_engine = torch_tensorrt.dynamo.compile(
        action_context_exported,
        inputs=action_context_input_specs,
        **ACTION_TRT_SETTINGS,
    )

    with torch.no_grad():
        trt_context_embs = action_context_trt_engine(trt_context_input) # logits, lm_hidden, prefix_k, prefix_v

    for _ in range(5):
        action_context_trt_engine(trt_context_input)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        action_context_trt_engine(trt_context_input)
    end.record()
    torch.cuda.synchronize()
    action_context_trt_elapsed_ms = start.elapsed_time(end) / 100

    parity("action context A vs C", eager_context_embs, trt_context_embs)

    # step 4 action 
    print("compiling diffusion")
    
    '''
    Flow-matching denoising is the same loop for every VLA.
    the StaticActionVelocityStep wrapper is one generic orchestrator:

    for each denoise step:
        build inputs from (noisy actions, timestep, context)   ← MODEL-SPECIFIC
        run the action expert (DiT / Gemma / …)                 ← MODEL-SPECIFIC module, GENERIC call
        pick action-token hidden states                         ← mostly generic
        decode hidden → velocity                                ← MODEL-SPECIFIC module, GENERIC call
        x = x + dt * velocity                                    ← generic (Euler), lives outside

    gr00t one step:
    inputs: x_t[B,50,32], timestep[B], vl_embs[B,S,1536], state[B,1,64], embodiment_id[B]
        │
        ▼ GrootDiTStepEncoder.forward
            state_encoder(state, emb)        → state_features
            action_encoder(x_t, t, emb)      → action_features
            cat(state, future_tokens, action)→ sa_embs
            returns expert_kwargs{hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep}
                    + decoder_args=(embodiment_id,)
        │
        ▼ action_expert = DiT(sa_embs, vl_embs, timestep)   → model_output [B, T, inner_dim]
        │
        ▼ get_action_hidden → keep last output_tokens (=50) → action_hidden [B, 50, inner_dim]
        │
        ▼ velocity_decoder(action_hidden, embodiment_id)    → velocity [B, 50, 32]
        │
        ▼ process_velocity (identity for GR00T)             → velocity
    '''
    
    diffusion_model = StaticActionVelocityStepExportModule(
        step_encoder=GrootDiTStepEncoderExportModule(model.action_head, embodiment_id),
        action_expert=model.action_head.model,
        velocity_decoder=TRTDynamicCategorySpecificMLPExportModule(
            model.action_head.action_decoder
        ),
        output_tokens=model.action_head.config.action_horizon,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

    '''# make action inputs 
    action_horizon = model.action_head.config.action_horizon
    action_dim = model.action_head.config.action_dim
    action_batch = trt_context_embs.shape[0]   # 1

    actions = torch.randn(
        action_batch,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )

    timestep = torch.zeros(
        action_batch,
        device=device,
        dtype=dtype,
    )

    eager_diffusion_input = (
        actions,
        timestep,
        eager_context_embs,
        state,
        embodiment_id,
    )

    trt_diffusion_input = (
        actions,
        timestep,
        trt_context_embs,
        state,
        embodiment_id,
    )

    diffusion_exported = torch.export.export(diffusion_model, args=trt_diffusion_input, strict=False)
    diffusion_input_specs = make_input_spec(trt_diffusion_input)
    diffusion_trt_engine = torch_tensorrt.dynamo.compile(
        diffusion_exported,
        inputs=diffusion_input_specs,
        **ACTION_TRT_SETTINGS,
    )

    seed = 42
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    eager_actions = torch.randn(
        action_batch,
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=dtype,
        generator=generator
    )
    trt_actions = torch.randn(
        action_batch,
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=dtype,
        generator=generator
    )

    num_steps = model.action_head.num_inference_timesteps

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        timestep_bucket = int(t_cont * model.action_head.num_timestep_buckets)

        timestep = torch.full(
            (actions.shape[0],),
            timestep_bucket,
            device=device,
            dtype=dtype,
        )

        eager_runner_inputs = (
            eager_actions,
            timestep,
            eager_context_embs,
            state,
            embodiment_id,
        )

        eager_out = diffusion_model(
            eager_actions,
            timestep,
            eager_context_embs,
            state,
            embodiment_id,
        )[0].to(dtype=dtype)
        dt = 1.0 / num_steps
        eager_actions = eager_actions + dt * eager_out

        trt_out = diffusion_trt_engine(
            trt_actions,
            timestep,
            trt_context_embs,
            state,
            embodiment_id,
        )[0].to(dtype=dtype)
        dt = 1.0 / num_steps
        trt_actions = trt_actions + dt * trt_out

    parity("action A vs C", eager_actions, trt_actions)
    '''

    # ---------------------------------------------------------------------------
    # Isolate diffusion engine parity: same inputs, one denoising step only.
    # This checks StaticActionVelocityStep eager vs TRT without rollout accumulation
    # or eager_context vs trt_context differences.
    # ---------------------------------------------------------------------------
    action_horizon = model.action_head.config.action_horizon
    action_dim = model.action_head.config.action_dim
    action_batch = trt_context_embs.shape[0]   # 1

    actions = torch.randn(
        action_batch,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )

    step_actions = actions.clone().to(device=device, dtype=dtype).contiguous()
    step_timestep = torch.zeros(
        step_actions.shape[0],
        device=device,
        dtype=torch.long,
    )
    # we must use same context embds input to isolate diffusion
    step_context = trt_context_embs.to(device=device, dtype=dtype).contiguous()
    
    diffusion_input = (
        step_actions,
        step_timestep,
        step_context,
        state,
        embodiment_id,
    )

    diffusion_exported = torch.export.export(diffusion_model, args=diffusion_input, strict=False)
    diffusion_input_specs = make_input_spec(diffusion_input)
    diffusion_trt_engine = torch_tensorrt.dynamo.compile(
        diffusion_exported,
        inputs=diffusion_input_specs,
        **ACTION_TRT_SETTINGS,
    )

    with torch.no_grad():
        eager_velocity = diffusion_model(
            step_actions,
            step_timestep,
            step_context,
            state,
            embodiment_id,
        )[0]

        trt_velocity = diffusion_trt_engine(
            step_actions,
            step_timestep,
            step_context,
            state,
            embodiment_id,
        )[0]

    for _ in range(5):
        diffusion_model(
            step_actions,
            step_timestep,
            step_context,
            state,
            embodiment_id,
        )[0]

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        diffusion_model(
            step_actions,
            step_timestep,
            step_context,
            state,
            embodiment_id,
        )[0]
    end.record()
    torch.cuda.synchronize()
    diffusion_eager_elapsed_ms = start.elapsed_time(end) / 100

    for _ in range(5):
        diffusion_trt_engine(
            step_actions,
            step_timestep,
            step_context,
            state,
            embodiment_id,
        )[0]

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        diffusion_trt_engine(
            step_actions,
            step_timestep,
            step_context,
            state,
            embodiment_id,
        )[0]
    end.record()
    torch.cuda.synchronize()
    diffusion_trt_elapsed_ms = start.elapsed_time(end) / 100

    parity("diffusion step A vs C", eager_velocity, trt_velocity)

    eager_total_ms = (
        vision_eager_elapsed_ms
        + eager_elapsed_ms
        + action_context_eager_elapsed_ms
        + diffusion_eager_elapsed_ms
    )
    trt_total_ms = (
        vision_trt_elapsed_ms
        + trt_elapsed_ms
        + action_context_trt_elapsed_ms
        + diffusion_trt_elapsed_ms
    )

    print(f"vision eager execute: {vision_eager_elapsed_ms:.3f} ms")
    print(f"vision trt execute: {vision_trt_elapsed_ms:.3f} ms")
    print(f"vision speedup: {(vision_eager_elapsed_ms / vision_trt_elapsed_ms):.3f}x")
    print(f"lm eager execute: {eager_elapsed_ms:.3f} ms")
    print(f"lm trt execute: {trt_elapsed_ms:.3f} ms")
    print(f"lm speedup: {(eager_elapsed_ms / trt_elapsed_ms):.3f}x")
    print(f"action context eager execute: {action_context_eager_elapsed_ms:.3f} ms")
    print(f"action context trt execute: {action_context_trt_elapsed_ms:.3f} ms")
    print(f"action context speedup: {(action_context_eager_elapsed_ms / action_context_trt_elapsed_ms):.3f}x")
    print(f"diffusion eager execute: {diffusion_eager_elapsed_ms:.3f} ms")
    print(f"diffusion trt execute: {diffusion_trt_elapsed_ms:.3f} ms")
    print(f"diffusion speedup: {(diffusion_eager_elapsed_ms / diffusion_trt_elapsed_ms):.3f}x")
    print(f"total eager execute: {eager_total_ms:.3f} ms")
    print(f"total trt execute: {trt_total_ms:.3f} ms")
    print(f"total speedup: {(eager_total_ms / trt_total_ms):.3f}x")
    return 0

if __name__ == "__main__":
    SystemExit(main())