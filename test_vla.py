import torch
import argparse
import torch_tensorrt
import logging

torch_tensorrt.logging.set_level(logging.ERROR)

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep

from trt.modules.export.vision import GridVisionExportModule
from trt.modules.export.language import CausalLMExportModule, ContextProjectionExportModule

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
from trt.compile import _make_input_spec

from typing import Any

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    #"use_python_runtime": True,
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

def prepare_compile_inputs(
    self,
    *,
    data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pil_messages = create_pil_messages(data)
    return prepare_model_inputs(
        self.eagle_processor,
        self.eagle_processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            }
        },
        data,
        pil_messages,
        self.device,
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()
    
    dtype = torch.float16

    config, policy = load_config(device)
    model = policy._groot_model
    eagle = model.backbone.eagle_model
    vision = model.backbone.eagle_model.vision_model
    language = model.backbone.eagle_model.language_model
    select_layer = model.backbone.select_layer

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

    pil_messages = create_pil_messages(data)
    model_inputs = prepare_model_inputs(
        eagle_processor,
        eagle_processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            }
        },
        data,
        pil_messages,
        device,
    )
    
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = tokenized_data["attention_mask"].to(device=device, dtype=torch.long)
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=dtype)
    state = pack_state(
        model_inputs["state"],  # [7] libero proprio
        max_state_dim=64,  # 64
        device=device,
    ) 
    action_side = {
        "state": state,
        "embodiment_id": make_embodiment_id(policy, state, device),
    }
    
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
        input_specs = _make_input_spec((pixel_values,))
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            embs_trt = trt_engine(pixel_values)
    finally:
        # always undo the patch so later eager runs aren't affected
        from trt.plugin.plugin_utils import restore_attention
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

    flat_image_embs = embs_eager.reshape(-1, hidden).to(
        device=flat_embs.device,
        dtype=flat_embs.dtype,   # match embedding dtype (bf16), not hardcoded fp16
    )
    flat_embs[image_token_mask] = flat_image_embs[:int(image_token_mask.sum().item())]
    inputs_embeds = flat_embs.reshape(bsz, seq_len, hidden)
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # ---------------------------------------------------------------------------
    # RUNG A: eager language — no wrapper, no patch
    # HF owns RoPE, KV cache, layer loop, norm
    # ---------------------------------------------------------------------------
    with torch.no_grad():
        eager_out = language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    lm_hidden_eager = eager_out.hidden_states[-1]

    # pre action transformation - action context
    eager_context_embs = lm_hidden_eager
    eager_context_embs = model.backbone.eagle_linear(eager_context_embs)

    # Match action_head.process_backbone_output().
    vlln_weight = getattr(model.action_head.vlln, "weight", None)
    if vlln_weight is not None:
        eager_context_embs = eager_context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
    eager_context_embs = model.action_head.vlln(eager_context_embs)
    eager_context_embs = model.action_head.vl_self_attention(eager_context_embs)

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
        select_layer=-1,
    ).eval().to(device=device)

    lm_inputs = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    pad_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
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
        lm_inputs,
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
        enable_bidirectional_prefill=0,
    )

    try:
        with torch.no_grad():
            logits, lm_hidden_trt_ref, prefix_k, prefix_v = lm(*flat_tensors)
            # lm_hidden_trt_ref is what CausalLMExportModule calls context_hidden
            # compare lm_hidden_eager vs lm_hidden_trt_ref (after TRT compile, same flat_tensors in)
    
        lm_exported = torch.export.export(lm, args=flat_tensors, strict=False)
        lm_input_specs = _make_input_spec(flat_tensors)
        lm_trt_engine = torch_tensorrt.dynamo.compile(
            lm_exported,
            inputs=lm_input_specs,
            **{**LANGUAGE_TRT_SETTINGS, "use_python_runtime": True},
        )

        with torch.no_grad():
            trt_out = lm_trt_engine(*flat_tensors)
            # trt_out is tuple: (logits, context_hidden, prefix_k, prefix_v)

    finally:
        restore_attention(patched)

    parity("language A vs C", lm_hidden_eager, trt_out[1])

    action_context = ContextProjectionExportModule(
        model.backbone.eagle_linear,
        model.action_head.vlln,
        model.action_head.vl_self_attention,
    ).eval()

    try:
        with torch.no_grad():
            # context hidden as input
            context = action_context(trt_out[1])

        action_context_exported = torch.export.export(action_context, args=(trt_out[1],), strict=False)
        action_context_input_specs = _make_input_spec((trt_out[1],))
        action_context_trt_engine = torch_tensorrt.dynamo.compile(
            action_context_exported,
            inputs=action_context_input_specs,
            **ACTION_TRT_SETTINGS,
        )

        with torch.no_grad():
            trt_context_embs = action_context_trt_engine(trt_out[1])
            # trt_out is tuple: (logits, context_hidden, prefix_k, prefix_v)

    finally:
        restore_attention(patched)

    parity("action context A vs C", eager_context_embs, trt_context_embs)
    return 0

def parity(name, a, b):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    rel_l2 = (a - b).norm() / b.norm().clamp_min(1e-8)
    rel_mean_pct = diff.mean() / b.abs().mean().clamp_min(1e-8) * 100
    close = torch.isclose(a, b, rtol=1e-2, atol=1e-2).float().mean() * 100
    print(
        f"{name:<22} mean_abs={diff.mean():.6f}  max_abs={diff.max():.6f}  "
        f"rel_l2={rel_l2:.4f}  rel_mean%={rel_mean_pct:.2f}  close%={close:.1f}"
    )
    
if __name__ == "__main__":
    SystemExit(main())