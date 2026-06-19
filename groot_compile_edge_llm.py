from __future__ import annotations

import os
import argparse
import pathlib
import subprocess
import copy
import json
import logging
import time

from typing import Any, Callable

import torch
import torch.nn as nn
import torch_tensorrt

from transformers import AutoProcessor

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.groot_n1 import DEFAULT_TOKENIZER_ASSETS_REPO
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.compile import (
    compile_trt_module, 
    dump_edge_fixture,
    save_trt_engine_module,
)

from trt.diffusion import StaticActionVelocityStep, GrootDiTStepEncoder, TRTDynamicCategorySpecificMLP
from trt.utils import (
    load_policy,
    compact_prefix_inputs,
    prepare_policy_inputs_groot,
)
from trt.helper import (
    get_processor
)
from trt.data import (
    load_test_data,
    prepare_model_inputs,
    pack_state
)
from trt.packing import (
    MultimodalPromptProcessor,
    PackedLanguageInputs,
    PromptPackingSpec,
    PromptTensorInputs,
)
from trt.vision import (
    VisualFixedInput, 
    PixelOnlyWrapper
)

from trt.language import (
    compile_language_trt_with_plugin,
    GROOTContextProjectionWrapper,
    GROOTLanguageContextWrapper,
    language_edge_llm_config,
    language_head_dim,
    make_groot_language_context_wrapper,
    FlatKVLanguageEngineWrapper
)
from trt.rope import (
    make_rope_rotary_cos_sin,
)
from trt.measure import (
    mean,
    std,
    print_timing,
    print_action_metrics,
    tensor_error_metrics,
    compute_action_parity_metrics

)
from trt.plugin_utils import (
    register_plugin_op,
    load_plugin,
    patch_vision_attention,  
    restore_attention,
    infer_siglip_seq_len,
    load_plugins_for_trt
)
from trt.io_spec import (
    GROOT_ACTION_ROLLOUT,
    GROOT_EDGE_IO,
    PipelineIOSpec,
    action_rollout_extra_config,
)
from trt.serialize import (
    SerializedTRTEngine,
    SerializedModuleSpec,
    load_serialized_modules,
    load_engine_config,
    SerializedGrootVision,
    SerializedGrootLanguage,
    SerializedGrootAction,
)

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    #"use_fp32_acc": True,
    "truncate_double": True,
    #"use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
    "use_fp32_acc": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "use_fp32_acc": True,
}

MODEL_ID = "nvidia/GR00T-N1.5-3B"
SEED = 42
WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GROOT_RUNTIME_BIN = WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt10/examples/groot/groot_runtime_smoke"

GROOT_EMBODIMENT_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}

logger = logging.getLogger(__name__)

@torch.no_grad()
def make_compile_inputs(action_horizon, action_dim, vl_embs, state, embodiment_id, device):
    batch_size = vl_embs.shape[0]
    dtype = vl_embs.dtype

    actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )

    timestep = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.long,
    )

    return (
        actions,
        timestep,
        vl_embs.to(device=device, dtype=dtype),
        state.to(device=device, dtype=dtype),
        embodiment_id.to(device=device),
    )

def make_static_action_module(
    action_head,
    device,
    dtype=torch.float16,
    embodiment_id=None,
):
    velocity_decoder = action_head.action_decoder

    if embodiment_id is not None:
        velocity_decoder = TRTDynamicCategorySpecificMLP(
            action_head.action_decoder
        )

    return StaticActionVelocityStep(
        step_encoder=GrootDiTStepEncoder(action_head, embodiment_id),
        action_expert=action_head.model,
        velocity_decoder=velocity_decoder,
        output_tokens=action_head.config.action_horizon,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

@torch.no_grad()
def build_language_inputs(core, vit_embs, input_ids, attention_mask=None) -> PackedLanguageInputs:
    eagle = core.backbone.eagle_model
    image_token_index = getattr(
        eagle,
        "image_token_index",
        eagle.config.image_token_index,
    )

    processor = MultimodalPromptProcessor(
        PromptPackingSpec(
            style="chat_template_placeholder",
            token_embed_fn=eagle.language_model.get_input_embeddings(),
            image_token_id=image_token_index,
        )
    )

    return processor(
        PromptTensorInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embs=vit_embs,
        )
    )

@torch.no_grad()
def build_context_from_language_inputs(core, packed: PackedLanguageInputs):
    eagle = core.backbone.eagle_model

    out = eagle.language_model(
        inputs_embeds=packed.inputs_embeds,
        attention_mask=packed.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    context_embs = out.hidden_states[core.backbone.select_layer]
    context_embs = core.backbone.eagle_linear(context_embs)

    vlln_weight = getattr(core.action_head.vlln, "weight", None)
    if vlln_weight is not None:
        context_embs = context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
    context_embs = core.action_head.vlln(context_embs)
    context_embs = core.action_head.vl_self_attention(context_embs)
    return context_embs


@torch.no_grad()
def build_context_inputs(core, vit_embs, input_ids, attention_mask):
    eagle = core.backbone.eagle_model
    packed = build_language_inputs(
        core,
        vit_embs,
        input_ids,
        attention_mask,
    )

    out = eagle.language_model(
        inputs_embeds=packed.inputs_embeds,
        attention_mask=packed.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    context_embs = out.hidden_states[core.backbone.select_layer]
    context_embs = core.backbone.eagle_linear(context_embs)

    # Match action_head.process_backbone_output().
    vlln_weight = getattr(core.action_head.vlln, "weight", None)
    if vlln_weight is not None:
        context_embs = context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
    context_embs = core.action_head.vlln(context_embs)
    context_embs = core.action_head.vl_self_attention(context_embs)

    return (
        context_embs,
        packed.pad_mask,
        packed.attention_mask,
        packed.position_ids,
    )

def make_context_masks(context_embs, attention_mask):
    context_pad_masks = attention_mask.to(device=context_embs.device, dtype=torch.bool)
    context_position_ids = torch.cumsum(context_pad_masks, dim=1) - 1

    return compact_prefix_inputs(
        context_embs,
        context_pad_masks,
        context_position_ids,
    )

def make_visual_fixed_input(
    model: nn.Module,
    sample_pixel_values: torch.Tensor,
    *,
    device,
    dtype=torch.float16,
) -> VisualFixedInput:
    eagle = model.backbone.eagle_model
    return VisualFixedInput(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=sample_pixel_values,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
    ).eval().to(device=device, dtype=dtype)

def save_visual_engine_for_edge_llm(
    model,
    pixel_values,
    engine_dir,
    *,
    device="cuda",
    dtype=torch.float16,
    model_type="vision",
    visual: nn.Module,
    io: PipelineIOSpec,
    trt_settings=None,
):
    pixel_values = pixel_values.to(device=device, dtype=dtype).contiguous()

    visual = make_visual_fixed_input(
        model,
        pixel_values,
        device=device,
        dtype=dtype,
    )

    vision_model = model.backbone.eagle_model.vision_model.vision_model

    with torch.no_grad():
        eager_output = visual(pixel_values)

    batch_size, seq_len = infer_siglip_seq_len(vision_model, pixel_values)

    patched = []
    try:
        patched = patch_vision_attention(
            vision_model,
            batch_size=batch_size,
            seq_len=seq_len,
            name="SigLIP",
        )

        return save_trt_engine_module(
            visual,
            (pixel_values,),
            engine_dir,
            engine_file="visual.engine",
            model_type=model_type,
            component="vision",
            input_names=list(io.vision.input_names),
            output_names=list(io.vision.output_names),
            example_output=eager_output,
            extra_config={
                "siglip_batch_size": batch_size,
                "siglip_seq_len": seq_len,
            },
            trt_settings=trt_settings,
        )

    finally:
        if patched:
            restore_attention(patched)

def save_lm_engine_for_edge_llm(
    core,
    input_embs,
    engine_dir,
    *,
    device,
    position_ids=None,
    dtype=torch.float16,
    model_type="language",
    io: PipelineIOSpec,
):
    max_seq_len = int(input_embs.shape[1])
    batch_size = int(input_embs.shape[0])
    input_embs = input_embs.to(device=device, dtype=dtype).contiguous()

    language_model = copy.deepcopy(core.backbone.eagle_model.language_model).to(
        device=device,
        dtype=torch.float16,
    ).eval()
    decoder = getattr(language_model, "model", language_model)
    cfg = language_model.config
    head_dim = language_head_dim(cfg)

    plugin_language = make_groot_language_context_wrapper(
        core,
        decoder,
        cfg,
        device=device,
        dtype=dtype,
        enable_bidirectional_prefill=0,
    )

    kv_caches = [
        torch.zeros(
            batch_size,
            2,  # key + value
            int(cfg.num_key_value_heads),
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(len(decoder.layers))
    ]

    ctx_len = torch.full(
        (batch_size,),
        max_seq_len,
        device=device,
        dtype=torch.int32,
    )

    wrapper = FlatKVLanguageEngineWrapper(plugin_language).to(device=device).eval()
    # Placeholder RoPE cache for export/compile tracing (values are ignored).
    rope_rotary_cos_sin = torch.randn(
        1,
        int(max_seq_len),
        int(head_dim),
        dtype=torch.float32,
        device=device,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (batch_size, 1),
        max_seq_len - 1,
        device=device,
        dtype=torch.int64,
    )
    sample_inputs = (
        input_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )
    input_names = io.language_input_names(len(kv_caches))

    with torch.no_grad():
        example_logits, example_context_embs = wrapper(*sample_inputs)

    return save_trt_engine_module(
        wrapper,
        sample_inputs,
        engine_dir,
        engine_file="language.engine",
        model_type=model_type,
        component="language",
        input_names=input_names,
        output_names=list(io.language.output_names),
        dual_optimization_profiles=True,
        example_output=(example_logits, example_context_embs),
        extra_config=language_edge_llm_config(
            cfg,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_layers=len(kv_caches),
            context_hidden_size=int(example_context_embs.shape[2]),
        ),
    )

def save_action_diffusion_engine_for_edge_llm(
    core,
    context_embs,
    state,
    embodiment_id,
    engine_dir,
    *,
    device,
    dtype=torch.float16,
    model_type="action",
    io: PipelineIOSpec,
):
    action_module = make_static_action_module(core.action_head, device, dtype, embodiment_id)

    context_embs = context_embs.to(device=device, dtype=dtype).contiguous()
    state = state.to(device=device, dtype=dtype).contiguous()
    embodiment_id = embodiment_id.to(device=device).contiguous()

    sample_inputs = make_compile_inputs(
        core.action_head.config.action_horizon,
        core.action_head.config.action_dim,
        context_embs,
        state,
        embodiment_id,
        device,
    )

    sample_inputs = tuple(
        x.contiguous() if isinstance(x, torch.Tensor) else x
        for x in sample_inputs
    )

    with torch.no_grad():
        eager_output = action_module(*sample_inputs)

    cfg = core.action_head.config

    return save_trt_engine_module(
        action_module,
        sample_inputs,
        engine_dir,
        engine_file="diffusion.engine",
        model_type=model_type,
        component="diffusion",
        input_names=list(io.action.input_names),
        output_names=list(io.action.output_names),
        example_output=eager_output,
        extra_config={
            "engine_role": "single_action_denoising_step",
            **action_rollout_extra_config(
                io,
                GROOT_ACTION_ROLLOUT,
                num_steps=int(core.action_head.num_inference_timesteps),
                num_timestep_buckets=int(core.action_head.num_timestep_buckets),
                action_horizon=int(cfg.action_horizon),
                action_dim=int(cfg.action_dim),
                context_seq_len=int(context_embs.shape[1]),
                context_hidden_size=int(context_embs.shape[2]),
                state_horizon=int(state.shape[1]),
                state_dim=int(state.shape[2]),
            ),
        },
    )

@torch.no_grad()
def run_serialized_groot_language(
    engine_lm: SerializedGrootLanguage,
    model: nn.Module,
    language_inputs: PackedLanguageInputs,
    device: torch.device,
) -> torch.Tensor:
    """Run an exported language.engine and return context_embs."""
    lm_inputs = language_inputs.inputs_embeds.to(device=device, dtype=torch.float16)
    language_model = model.backbone.eagle_model.language_model
    decoder = getattr(language_model, "model", language_model)
    cfg = language_model.config
    max_seq_len = int(lm_inputs.shape[1])
    language_head_dim_val = language_head_dim(cfg)

    kv_caches = [
        torch.zeros(
            int(lm_inputs.shape[0]),
            2,  # key + value
            int(cfg.num_key_value_heads),
            max_seq_len,
            language_head_dim_val,
            device=device,
            dtype=lm_inputs.dtype,
        )
        for _ in range(len(decoder.layers))
    ]
    ctx_len = torch.full(
        (lm_inputs.shape[0],),
        lm_inputs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        max_seq_len,
        device,
        language_model=language_model,
        position_ids=language_inputs.position_ids,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (int(lm_inputs.shape[0]), 1),
        int(lm_inputs.shape[1]) - 1,
        device=device,
        dtype=torch.int64,
    )
    lm_out = engine_lm(
        lm_inputs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        kv_caches,
    )
    if isinstance(lm_out, tuple):
        return lm_out[1]
    return lm_out

def save_edge_engines_for_edge_llm(
    model: nn.Module,
    policy: Any,
    device: str,
    model_inputs: dict,
    *,
    seed: int = 42,
    offload_module_to_cpu: bool = False,
    max_generation_length: int = 256,
    num_traj_samples: int = 1,
    max_seq_len: int | None = None,
    debug: bool = False,
    accuracy_check: bool = True,
    engine_root: str,
    io: PipelineIOSpec,
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, dict]:
    engine_root = str(pathlib.Path(engine_root))
    tokenized_data = model_inputs['tokenized_data']
    input_ids = tokenized_data['input_ids']

    # groot specifc inputs ------
    attention_mask = tokenized_data['attention_mask']
    state, state_mask = pack_state(
        model_inputs["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    embodiment_id = torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

    # Keep the raw image pixels as a one-stream list so this mirrors the PI0.5 script.
    images = [tokenized_data["pixel_values"].to(
        device=device,
        dtype=torch.float16,
    )]
    pixel_values = images[0]
    # groot specifc inputs ------

    # Load the custom TensorRT plugin library before compiling plugin-backed modules.
    load_plugins_for_trt()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    visual = make_visual_fixed_input(
        model,
        pixel_values,
        device=device,
        dtype=torch.float16,
    )

    engine_dir = str(pathlib.Path(engine_root) / "visual")
    save_visual_engine_for_edge_llm(
        model,
        pixel_values,
        engine_dir,
        device=device,
        dtype=torch.float16,
        visual=visual,
        io=io,
        trt_settings=VISION_TRT_SETTINGS,
    )

    vision_runner = SerializedGrootVision(SerializedTRTEngine(engine_dir))
    with torch.no_grad():
        trt_image_embs = vision_runner(pixel_values)

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")

    language_inputs = build_language_inputs(
        model,
        trt_image_embs,
        input_ids,
        attention_mask,
    )

    language_engine_dir = str(pathlib.Path(engine_root) / "language")
    save_lm_engine_for_edge_llm(
        model,
        language_inputs.inputs_embeds,
        language_engine_dir,
        device=device,
        position_ids=None,
        dtype=torch.float16,
        io=io,
    )

    language_runner = SerializedGrootLanguage(SerializedTRTEngine(language_engine_dir))
    with torch.no_grad():
        context_embs = run_serialized_groot_language(
            language_runner,
            model,
            language_inputs,
            torch.device(device),
        )

    # -------------------------
    # Action/diffusion engine
    # -------------------------
    print("compiling action diffusion")

    action_engine_dir = str(pathlib.Path(engine_root) / "action")
    trt_diffusion = save_action_diffusion_engine_for_edge_llm(
        model,
        context_embs,
        state,
        embodiment_id,
        action_engine_dir,
        device=device,
        dtype=torch.float16,
        io=io,
    )

    fixture_dir = _dump_groot_edge_fixture(
        engine_root=engine_root,
        model=model,
        policy=policy,
        pixel_values=pixel_values,
        input_ids=input_ids,
        language_inputs=language_inputs,
        visual_embeds=trt_image_embs,
        context_embs=context_embs,
        state=state,
        embodiment_id=embodiment_id,
        seed=seed,
        device=device,
    )

    plugin_info = {
        "engine_root": engine_root,
        "vision_engine_dir": str(pathlib.Path(engine_root) / "visual"),
        "language_engine_dir": language_engine_dir,
        "action_engine_dir": action_engine_dir,
        "vision_engine": str(pathlib.Path(engine_dir) / "visual.engine"),
        "language_engine": str(pathlib.Path(language_engine_dir) / "language.engine"),
        "diffusion_engine": str(pathlib.Path(action_engine_dir) / "diffusion.engine"),
        "language_seq_len": int(language_inputs.inputs_embeds.shape[1]),
        "context_seq_len": int(context_embs.shape[1]),
        "context_hidden_size": int(context_embs.shape[2]),
        "state_shape": list(state.shape),
        "embodiment_id": embodiment_id.detach().cpu().tolist(),
        "fixture_dir": str(fixture_dir),
        **io.to_plugin_info(),
    }

    return None, None, trt_diffusion, plugin_info


def compile_trt_with_plugin(
    model: nn.Module,
    policy: Any,
    device: torch.device,
    model_inputs: dict,
    *,
    seed: int = 42,
    offload_module_to_cpu: bool = False,
    max_generation_length: int = 256,
    num_traj_samples: int = 1,
    max_seq_len: int | None = None,
    debug: bool = False,
    accuracy_check: bool = True,
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, dict]:
    '''
    tokenized_data
    input_ids          text token IDs, including image placeholder tokens
    attention_mask     text mask
    pixel_values       processed image tensor
    '''
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    # GROOT is built to handle multiple embodiments, 
    # where each robot can expose a different state vector width so 
    # GROOT uses a fixed max_state_dim.
    state, _ = pack_state(
        model_inputs["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )
    state = state.to(device=device, dtype=torch.float16).contiguous()
    # the same model can support different robots but the action head has robot-specific weights.
    # embodiment_id chooses which embodiment-specific state/action encoder and decoder weights to use
    embodiment_id = _make_embodiment_id(policy, state, device).contiguous()
    # pixelk values [B, C, H, W]
    pixel_values = tokenized_data["pixel_values"].to(
        device=device,
        dtype=torch.float16,
    ).contiguous()

    load_plugins_for_trt()

    plugin_settings = {
        **TRT_SETTINGS,
        "use_python_runtime": True,
        "use_fp32_acc": True,
    }
    vision_settings = {
        **VISION_TRT_SETTINGS,
        "use_python_runtime": True,
    }
    action_settings = {
        **ACTION_TRT_SETTINGS,
        "use_python_runtime": True,
    }

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    visual = make_visual_fixed_input(
        model,
        pixel_values,
        device=device,
        dtype=torch.float16,
    )

    with torch.no_grad():
        eager_image_embs = visual(pixel_values)

    vision_model = model.backbone.eagle_model.vision_model.vision_model
    batch_size, seq_len = infer_siglip_seq_len(vision_model, pixel_values)

    patched = []
    try:
        patched = patch_vision_attention(
            vision_model,
            batch_size=batch_size,
            seq_len=seq_len,
            name="SigLIP",
        )
        trt_vision = compile_trt_module(
            visual,
            (pixel_values,),
            vision_settings,
        )
    finally:
        if patched:
            restore_attention(patched)

    with torch.no_grad():
        trt_image_embs = trt_vision(pixel_values)

    if accuracy_check:
        tensor_error_metrics("groot TRT vs original vision embeddings", trt_image_embs, eager_image_embs)

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")

    language_inputs = build_language_inputs(
        model,
        trt_image_embs,
        input_ids,
        attention_mask,
    )

    language_max_seq_len = int(language_inputs.inputs_embeds.shape[1])
    if max_seq_len is not None:
        language_max_seq_len = int(max_seq_len)

    language_model = copy.deepcopy(model.backbone.eagle_model.language_model).to(
        device=device,
        dtype=torch.float16,
    ).eval()
    decoder = getattr(language_model, "model", language_model)
    cfg = language_model.config

    plugin_language = make_groot_language_context_wrapper(
        model,
        decoder,
        cfg,
        device=device,
        dtype=torch.float16,
        enable_bidirectional_prefill=0,
    )

    trt_lm, trt_max_seq_len = compile_language_trt_with_plugin(
        plugin_language,
        language_inputs.inputs_embeds,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        device=device,
        settings=plugin_settings,
        max_seq_len=language_max_seq_len,
    )
    language_max_seq_len = int(trt_max_seq_len)
    language_head_dim_val = language_head_dim(cfg)

    with torch.no_grad():
        lm_inputs = language_inputs.inputs_embeds.to(device=device, dtype=torch.float16)
        kv_caches = [
            torch.zeros(
                int(lm_inputs.shape[0]),
                2,  # key + value
                int(cfg.num_key_value_heads),
                language_max_seq_len,
                language_head_dim_val,
                device=device,
                dtype=lm_inputs.dtype,
            )
            for _ in range(len(decoder.layers))
        ]
        ctx_len = torch.full(
            (lm_inputs.shape[0],),
            lm_inputs.shape[1],
            device=device,
            dtype=torch.int32,
        )
        rope_rotary_cos_sin = make_rope_rotary_cos_sin(
            cfg,
            language_max_seq_len,
            device,
            language_model=language_model,
            position_ids=language_inputs.position_ids,
        )
        kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
        last_token_ids = torch.full(
            (int(lm_inputs.shape[0]), 1),
            int(lm_inputs.shape[1]) - 1,
            device=device,
            dtype=torch.int64,
        )
        _, trt_context_embs = trt_lm(
            lm_inputs,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            last_token_ids,
            kv_caches,
        )
        trt_context_embs = trt_context_embs.to(device=device, dtype=torch.float16)

        if accuracy_check:
            eager_context_embs, _, _, _ = build_context_inputs(
                model,
                trt_image_embs,
                input_ids,
                attention_mask,
            )
            tensor_error_metrics(
                "groot TRT vs eager language context (TRT vision)",
                trt_context_embs,
                eager_context_embs.to(device=device, dtype=torch.float16),
            )

    # -------------------------
    # Action/diffusion engine
    # -------------------------
    print("compiling action diffusion")

    action_module = make_static_action_module(
        model.action_head,
        device,
        torch.float16,
        embodiment_id,
    )

    sample_inputs = make_compile_inputs(
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        trt_context_embs,
        state,
        embodiment_id,
        device,
    )

    trt_diffusion = compile_trt_module(
        action_module,
        sample_inputs,
        action_settings,
    )

    action_module = action_module.to(device=device, dtype=torch.float16).eval()

    plugin_info = {
        "language_max_seq_len": language_max_seq_len,
        "language_head_dim": language_head_dim_val,
        "context_seq_len": int(trt_context_embs.shape[1]),
        "context_hidden_size": int(trt_context_embs.shape[2]),
        "state_shape": list(state.shape),
        "embodiment_id": embodiment_id.detach().cpu().tolist(),
    }

    return trt_vision, trt_lm, trt_diffusion, plugin_info

@torch.no_grad()
def run_inference_pytorch_groot(
    model,
    policy,
    create_inputs_fn: Callable[[], dict],
    *,
    seed: int,
    device: torch.device,
    vision_module=None,
) -> tuple[torch.Tensor, dict, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_inputs = create_inputs_fn()
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=torch.float16)

    state, _ = pack_state(
        model_inputs["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )
    state = state.to(device=device, dtype=torch.float16)

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    embodiment_id = torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

    start_time = time.perf_counter()

    with torch.autocast("cuda", dtype=torch.float16):
        if vision_module is None:
            image_embs = make_visual_fixed_input(
                model,
                pixel_values,
                device=device,
                dtype=torch.float16,
            )(pixel_values)
        else:
            image_embs = vision_module(pixel_values)

        context_embs, _, _, _ = build_context_inputs(
            model,
            image_embs,
            input_ids,
            attention_mask,
        )

        context_embs = context_embs.to(dtype=torch.float16)

        noise = torch.randn(
            context_embs.shape[0],
            model.action_head.config.action_horizon,
            model.action_head.config.action_dim,
            device=device,
            dtype=context_embs.dtype,
        )

        action_module = make_static_action_module(
            model.action_head,
            device,
            torch.float16,
            embodiment_id,
        )

        context = ActionRolloutContext(
            noise=noise,
            device=device,
            context_embs=context_embs,
            state=state,
            embodiment_id=embodiment_id,
        )

        actions = sample_actions_raw(
            action_module,
            context,
            GROOTActionAdapter(model.action_head),
        )

    elapsed = time.perf_counter() - start_time

    extra = {
        "noise": noise,
        "context_embs": context_embs,
        "state": state,
        "embodiment_id": embodiment_id,
    }

    return actions, extra, elapsed

@torch.no_grad()
def run_inference_trt_plugin(
    model,
    policy,
    create_inputs_fn: Callable[[], dict],
    *,
    trt_vision,
    trt_lm,
    trt_diffusion,
    plugin_info: dict,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_inputs = create_inputs_fn()
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=torch.float16).contiguous()

    state, _ = pack_state(
        model_inputs["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )
    state = state.to(device=device, dtype=torch.float16).contiguous()
    embodiment_id = _make_embodiment_id(policy, state, device).contiguous()

    start_time = time.perf_counter()

    image_embs = trt_vision(pixel_values)

    language_inputs = build_language_inputs(
        model,
        image_embs,
        input_ids,
        attention_mask,
    )

    lm_inputs = language_inputs.inputs_embeds.to(device=device, dtype=torch.float16)
    language_model = model.backbone.eagle_model.language_model
    decoder = getattr(language_model, "model", language_model)
    cfg = language_model.config
    kv_caches = [
        torch.zeros(
            int(lm_inputs.shape[0]),
            2,  # key + value
            int(cfg.num_key_value_heads),
            int(plugin_info["language_max_seq_len"]),
            language_head_dim(cfg),
            device=device,
            dtype=lm_inputs.dtype,
        )
        for _ in range(len(decoder.layers))
    ]
    ctx_len = torch.full(
        (lm_inputs.shape[0],),
        lm_inputs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        int(plugin_info["language_max_seq_len"]),
        device,
        language_model=language_model,
        position_ids=language_inputs.position_ids,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (int(lm_inputs.shape[0]), 1),
        int(lm_inputs.shape[1]) - 1,
        device=device,
        dtype=torch.int64,
    )

    lm_out = trt_lm(
        lm_inputs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        kv_caches,
    )
    if isinstance(lm_out, tuple):
        context_embs = lm_out[1]
    else:
        context_embs = lm_out
    context_embs = context_embs.to(device=device, dtype=torch.float16)

    noise = torch.randn(
        context_embs.shape[0],
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=context_embs.dtype,
    )

    actions = sample_actions_raw(
        trt_diffusion,
        ActionRolloutContext(
            noise=noise,
            device=device,
            context_embs=context_embs,
            state=state,
            embodiment_id=embodiment_id,
        ),
        GROOTActionAdapter(model.action_head),
    )

    elapsed = time.perf_counter() - start_time

    extra = {
        "noise": noise,
        "visual_embeds": image_embs,
        "context_embs": context_embs,
        "state": state,
        "embodiment_id": embodiment_id,
    }

    return actions, extra, elapsed


def _dump_groot_edge_fixture(
    *,
    engine_root: str,
    model: nn.Module,
    policy: Any,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    language_inputs: PackedLanguageInputs,
    visual_embeds: torch.Tensor,
    context_embs: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    seed: int,
    device: torch.device,
) -> pathlib.Path:
    eagle = model.backbone.eagle_model
    text_embeds = eagle.language_model.get_input_embeddings()(
        input_ids.to(device=device)
    ).to(device=device, dtype=torch.float16)

    state = state.to(device=device, dtype=torch.float16).contiguous()
    context_embs = context_embs.to(device=device, dtype=torch.float16).contiguous()
    embodiment_id = embodiment_id.to(device=device).contiguous()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial_actions = torch.randn(
        context_embs.shape[0],
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=context_embs.dtype,
        generator=generator,
    )
    timestep = torch.zeros(
        context_embs.shape[0],
        device=device,
        dtype=torch.long,
    )

    action_module = make_static_action_module(
        model.action_head,
        device,
        torch.float16,
        embodiment_id,
    )
    pred_velocity = action_module(
        initial_actions,
        timestep,
        context_embs,
        state,
        embodiment_id,
    )
    actions_out = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=initial_actions,
            device=device,
            context_embs=context_embs,
            state=state,
            embodiment_id=embodiment_id,
        ),
        GROOTActionAdapter(model.action_head),
    )

    if language_inputs.image_token_mask is None:
        raise ValueError("GR00T Edge fixture export requires image_token_mask for visual-to-LM packing")

    return dump_edge_fixture(
        engine_root,
        {
            "pixel_values": pixel_values.to(device=device, dtype=torch.float16),
            "text_embeds": text_embeds,
            "image_token_mask": language_inputs.image_token_mask.to(dtype=torch.uint8),
            "inputs_embeds": language_inputs.inputs_embeds.to(device=device, dtype=torch.float16),
            "visual_embeds": visual_embeds.to(device=device, dtype=torch.float16),
            "context_embs": context_embs,
            "state": state,
            "embodiment_id": embodiment_id,
            "initial_actions": initial_actions,
            "timestep": timestep,
            "pred_velocity": pred_velocity.to(device=device, dtype=torch.float16),
            "actions_out": actions_out.to(device=device, dtype=torch.float16),
        },
    )


def _make_embodiment_id(policy, state: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Build the per-sample robot/embodiment index consumed by GROOT's action head.

    GROOT supports multiple robot embodiments with different state/action layouts.
    The action head uses this integer ID to select embodiment-specific state
    encoder, action encoder, and action decoder weights. This script uses one
    robot per batch, so every batch element gets the same ID, but the tensor must
    still be shaped (B,) to match the model and TensorRT engine ABI.
    """
    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    return torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

def _groot_output_action_dim(policy) -> int | None:
    output_features = getattr(policy.config, "output_features", None)
    if output_features is None:
        return None

    action_feature = output_features.get(ACTION)
    if action_feature is None:
        return None

    shape = getattr(action_feature, "shape", None)
    if not shape:
        return None

    return int(shape[0])

def compute_groot_policy_action_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    policy,
) -> dict[str, float]:
    action_dim = _groot_output_action_dim(policy)
    if action_dim is not None:
        pred_actions = pred_actions[..., :action_dim]
        target_actions = target_actions[..., :action_dim]

    return compute_action_parity_metrics(pred_actions, target_actions)

def make_create_inputs_fn(processor, data, messages, device):
    def create_inputs():
        return prepare_model_inputs(
            processor,
            processor.process_vision_info,
            {"add_generation_prompt": True},
            {"images_kwargs": {"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False}},
            data,
            messages,
            device,
        )
    return create_inputs
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GR00T TensorRT engines for TensorRT-Edge-LLM")

    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="GR00T policy/model id to load.")
    parser.add_argument("--dataset-id", type=str, default="lerobot/libero", help="LeRobot dataset id used to build example compile inputs.")
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index used for the compile sample.")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index used for the compile sample.")

    parser.add_argument("--engine-dir", type=str, default="/tmp/groot_edge_llm", help="Root directory for exported Edge-LLM engines.")
    parser.add_argument("--groot-runtime-bin", type=str, default=str(DEFAULT_GROOT_RUNTIME_BIN), help="Path to the C++ groot_runtime_smoke executable.")
    parser.add_argument("--vision-engine-dir", type=str, default=None, help="Optional override for the vision engine directory.")
    parser.add_argument("--language-engine-dir", type=str, default=None, help="Optional override for the language engine directory.")

    parser.add_argument("--plugin-so", type=str, default=os.environ.get("EDGELLM_TRT_PLUGIN_SO") or os.environ.get("EDGE_LLM_PLUGIN_SO"), help="Path to libNvInfer_edgellm_plugin.so.")
    parser.add_argument("--device", type=str, default="cuda", help="Compile device.")

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for compile/test tensors.")
    parser.add_argument("--num-traj-samples", type=int, default=1, help="Number of GR00T trajectory samples for action/runtime checks.")
    parser.add_argument("--max-generation-length", type=int, default=256, help="Max language generation length for GR00T checks.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Optional static language sequence length override.")

    parser.add_argument("--skip-vision", action="store_true", help="Skip visual.engine export.")
    parser.add_argument("--skip-language", action="store_true", help="Skip language.engine export.")
    parser.add_argument("--skip-action", action="store_true", help="Skip action/diffusion engine export if enabled later.")

    parser.add_argument("--debug", action="store_true", help="Enable extra debug logging/checks.")
    parser.add_argument("--no-accuracy-check", action="store_true", help="Skip eager-vs-TRT accuracy checks.")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip eager PyTorch action rollout.")
    parser.add_argument("--skip-trt", action="store_true", help="Skip Python TRT plugin action rollout.")
    parser.add_argument("--skip-engine", action="store_true", help="Skip Python serialized .engine action rollout.")
    parser.add_argument("--no-stage-parity", action="store_true", help="Skip staged C++ vs eager parity diagnostics.")
    parser.add_argument("--num-iterations", type=int, default=12, help="Total timing iterations including warmup.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations to exclude from summary.")
    
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.plugin_so:
        os.environ["EDGELLM_TRT_PLUGIN_SO"] = args.plugin_so

    # load in episode 0, frame 0 using lerobot/libero dataset, frame 0 has 2 cameras (data is 2 still images)
    data, messages = load_test_data(
        dataset_id=args.dataset_id,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
    )

    cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
    processor = get_processor(
        str(cache_dir),
        {
            "trust_remote_code": True,
            "fix_mistral_regex": False,
        },
    )

    policy = load_policy(GrootPolicy, args.model_id, device).to(device).eval()
    model = policy._groot_model.to(device).eval()

    # TODO: right now we are using the same episode 0 and frame 0
    # each iteration should use a different frame
    create_inputs_fn = make_create_inputs_fn(
        processor,
        data,
        messages,
        device,
    )

    compile_inputs = create_inputs_fn()

    print(
        f"dataset={args.dataset_id}  episode={args.episode_index}  frame={args.frame_index}  "
        f"num_traj_samples={args.num_traj_samples}  iters={args.num_iterations}  warmup={args.warmup}"
    )

    trt_vision = trt_lm = trt_diffusion = plugin_info = None
    serialized_engine_info = None
    edge_plugin_info = None

    if not args.skip_trt:
        trt_vision, trt_lm, trt_diffusion, plugin_info = compile_trt_with_plugin(
            model,
            policy,
            device,
            compile_inputs,
            seed=args.seed,
            max_generation_length=args.max_generation_length,
            num_traj_samples=args.num_traj_samples,
            max_seq_len=args.max_seq_len,
            debug=args.debug,
            accuracy_check=not args.no_accuracy_check,
        )

    if not args.skip_engine:
        _, _, _, serialized_engine_info = save_edge_engines_for_edge_llm(
            model,
            policy,
            device,
            compile_inputs,
            seed=args.seed,
            max_generation_length=args.max_generation_length,
            num_traj_samples=args.num_traj_samples,
            max_seq_len=args.max_seq_len,
            debug=args.debug,
            accuracy_check=not args.no_accuracy_check,
            engine_root=args.engine_dir,
            io=GROOT_EDGE_IO,
        )
        edge_plugin_info = serialized_engine_info

    engine_vision = engine_lm = engine_diffusion = engine_info = None
    if not args.skip_engine:
        engine_vision, engine_lm, engine_diffusion, engine_info = load_serialized_modules(
            serialized_engine_info["engine_root"],
            specs=(
                SerializedModuleSpec("vision", "visual", SerializedGrootVision),
                SerializedModuleSpec("language", "language", SerializedGrootLanguage),
                SerializedModuleSpec("action", "action", SerializedGrootAction),
            ),
            plugin_info_aliases={
                "language_max_seq_len": ("language", "max_seq_len"),
                "context_seq_len": ("action", "context_seq_len"),
                "context_hidden_size": ("action", "context_hidden_size"),
            },
        )

    pt_times: list[float] = []
    trt_times: list[float] = []
    engine_times: list[float] = []
    action_ades: list[float] = []
    actionmean_abs: list[float] = []
    engine_action_ades: list[float] = []
    engine_actionmean_abs: list[float] = []

    pt_ref_for_trt = None
    if trt_vision is not None:
        pt_ref_for_trt, _, _ = run_inference_pytorch_groot(
            model,
            policy,
            create_inputs_fn,
            seed=args.seed,
            device=device,
            vision_module=trt_vision,
        )

    pt_ref_for_engine = None
    if engine_vision is not None:
        pt_ref_for_engine, _, _ = run_inference_pytorch_groot(
            model,
            policy,
            create_inputs_fn,
            seed=args.seed,
            device=device,
            vision_module=engine_vision,
        )

    for i in range(args.num_iterations):
        print(f"\n=== iter {i} ===", flush=True)

        pred_actions_pt = None

        # -- PyTorch eager -------------------------------------------------
        if not args.skip_pytorch:
            if device.type == "cuda":
                torch.cuda.synchronize(); t = time.perf_counter()
            else:
                t = time.perf_counter()
            pred_actions_pt, extra_pt, _ = run_inference_pytorch_groot(
                model,
                policy,
                create_inputs_fn,
                seed=args.seed,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            pt_elapsed = 1000 * (time.perf_counter() - t)
            pt_times.append(pt_elapsed)
            print(f"  PyTorch    : {pt_elapsed:7.1f} ms")

        # -- TRT Plugin FP16 ----------------------------------------------
        if not args.skip_trt:
            if device.type == "cuda":
                torch.cuda.synchronize(); t = time.perf_counter()
            else:
                t = time.perf_counter()
            pred_actions_trt, extra_trt, _ = run_inference_trt_plugin(
                model,
                policy,
                create_inputs_fn,
                trt_vision=trt_vision,
                trt_lm=trt_lm,
                trt_diffusion=trt_diffusion,
                plugin_info=plugin_info,
                seed=args.seed,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            trt_elapsed = 1000 * (time.perf_counter() - t)
            trt_times.append(trt_elapsed)

            if pt_ref_for_trt is not None:
                trt_metrics = compute_groot_policy_action_metrics(
                    pred_actions_trt,
                    pt_ref_for_trt,
                    policy,
                )
                action_ades.append(trt_metrics["action_ade"])
                actionmean_abs.append(trt_metrics["mean_abs"])
                print(f"  TRT Plugin : {trt_elapsed:7.1f} ms   actionADE={trt_metrics['action_ade']:.6f}  mean_abs={trt_metrics['mean_abs']:.6f}")
            else:
                print(f"  TRT Plugin : {trt_elapsed:7.1f} ms")

        # -- Serialized .engine --------------------------------------------
        if not args.skip_engine:
            if device.type == "cuda":
                torch.cuda.synchronize(); t = time.perf_counter()
            else:
                t = time.perf_counter()

            pred_actions_engine, extra_engine, _ = run_inference_trt_plugin(
                model,
                policy,
                create_inputs_fn,
                trt_vision=engine_vision,
                trt_lm=engine_lm,
                trt_diffusion=engine_diffusion,
                plugin_info=engine_info,
                seed=args.seed,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            engine_elapsed = 1000 * (time.perf_counter() - t)
            engine_times.append(engine_elapsed)

            if pt_ref_for_engine is not None:
                engine_metrics = compute_groot_policy_action_metrics(
                    pred_actions_engine,
                    pt_ref_for_engine,
                    policy,
                )
                engine_action_ades.append(engine_metrics["action_ade"])
                engine_actionmean_abs.append(engine_metrics["mean_abs"])
                print(f"  Serialized : {engine_elapsed:7.1f} ms   actionADE={engine_metrics['action_ade']:.6f}  mean_abs={engine_metrics['mean_abs']:.6f}")
            else:
                print(f"  Serialized : {engine_elapsed:7.1f} ms")

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations})")
    print("=" * 78)

    if pt_times:
        print_timing("PyTorch GR00T", pt_times[args.warmup:])

    if trt_times:
        print_timing("TRT Plugin FP16", trt_times[args.warmup:])

    if engine_times:
        print_timing("Serialized Engine", engine_times[args.warmup:])

    if action_ades:
        print_action_metrics("TRT Action ADE", action_ades[args.warmup:])
        print_action_metrics("TRT Action mean abs", actionmean_abs[args.warmup:])

    if engine_action_ades:
        print_action_metrics("Engine Action ADE", engine_action_ades[args.warmup:])
        print_action_metrics("Engine Action mean abs", engine_actionmean_abs[args.warmup:])

    if pt_times and trt_times:
        pt_avg = mean(pt_times[args.warmup:])
        trt_avg = mean(trt_times[args.warmup:])
        speedup = pt_avg / trt_avg if trt_avg > 0 else float("nan")
        print(f"\n  Speedup (TRT vs PyTorch): {speedup:5.2f}x   ({pt_avg:.1f} -> {trt_avg:.1f} ms)")

    if pt_times and engine_times:
        pt_avg = mean(pt_times[args.warmup:])
        engine_avg = mean(engine_times[args.warmup:])
        speedup = pt_avg / engine_avg if engine_avg > 0 else float("nan")
        print(f"  Speedup (Engine vs PyTorch): {speedup:5.2f}x   ({pt_avg:.1f} -> {engine_avg:.1f} ms)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())