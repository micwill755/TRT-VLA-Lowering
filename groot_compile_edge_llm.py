from __future__ import annotations

import os
import argparse
import ctypes
import pathlib
import subprocess
import torch
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
from lerobot.utils.constants import HF_LEROBOT_HOME

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.compile import (
    compile_trt_module, 
    save_trt_engine_module
)

from trt.diffusion import GrootStaticDiffusionStep
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
    make_batch,
    pack_state
)
from trt.packing import (
    MultimodalPromptProcessor,
    PackedLanguageInputs,
    PromptPackingSpec,
    PromptTensorInputs,
)
from trt.vision import (
    GROOTVisualFixedInput, 
    PixelOnlyWrapper
)

from trt.language import (
    compile_groot_lm_trt_with_plugin,
    make_groot_plugin_language,
    make_groot_language_kv_caches,
    run_groot_plugin_language,
    GROOTLanguageEngineWrapper
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
)
from trt.serialize import (
    SerializedTRTEngine,
    SerializedModuleSpec,
    load_serialized_modules,
    load_engine_config,
    SerializedGrootVision,
    SerializedGrootLanguage,
    SerializedGrootAction
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
    parser.add_argument("--skip-export", action="store_true", help="Skip Edge engine export.")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip eager PyTorch action rollout.")
    parser.add_argument("--skip-trt", action="store_true", help="Skip Python TRT plugin action rollout.")
    parser.add_argument("--skip-engine", action="store_true", help="Skip Python serialized .engine action rollout.")
    parser.add_argument("--skip-edge", action="store_true", help="Skip Edge/C++ runtime action rollout.")
    parser.add_argument("--no-stage-parity", action="store_true", help="Skip staged C++ vs eager parity diagnostics.")
    parser.add_argument("--num-iterations", type=int, default=12, help="Total timing iterations including warmup.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations to exclude from summary.")
    
    return parser.parse_args()

def make_compile_inputs(action_step, vl_embs, state, embodiment_id, device):
    batch_size = vl_embs.shape[0]
    dtype = vl_embs.dtype

    action_horizon = action_step.action_horizon
    action_dim = action_step.action_decoder.layer2.b.shape[-1]

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
        vl_embs,
        state,
        embodiment_id,
    )

@torch.no_grad()
def build_groot_language_inputs(core, vit_embs, input_ids, attention_mask=None) -> PackedLanguageInputs:
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
def build_groot_context_from_language_inputs(core, packed: PackedLanguageInputs):
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
def build_groot_context_inputs(core, vit_embs, input_ids, attention_mask):
    eagle = core.backbone.eagle_model
    packed = build_groot_language_inputs(
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

def make_groot_context_masks(context_embs, attention_mask):
    context_pad_masks = attention_mask.to(device=context_embs.device, dtype=torch.bool)
    context_position_ids = torch.cumsum(context_pad_masks, dim=1) - 1

    return compact_prefix_inputs(
        context_embs,
        context_pad_masks,
        context_position_ids,
    )

def save_groot_visual_engine_for_edge_llm(
    model,
    pixel_values,
    engine_dir,
    *,
    device="cuda",
    dtype=torch.float16,
    model_type="groot_vision",
):
    pixel_values = pixel_values.to(device=device, dtype=dtype).contiguous()

    visual = GROOTVisualFixedInput(
        model,
        pixel_values,
    ).eval().to(device=device, dtype=dtype)

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
            input_names=["pixel_values"],
            output_names=["visual_embeds"],
            example_output=eager_output,
            extra_config={
                "siglip_batch_size": batch_size,
                "siglip_seq_len": seq_len,
            },
        )

    finally:
        if patched:
            restore_attention(patched)

def save_groot_lm_engine_for_edge_llm(
    core,
    input_embs,
    engine_dir,
    *,
    device,
    position_ids=None,
    dtype=torch.float16,
    model_type="groot_language",
):
    max_seq_len = int(input_embs.shape[1])
    batch_size = int(input_embs.shape[0])

    plugin_language = make_groot_plugin_language(
        core,
        max_seq_len=max_seq_len,
        device=device,
        position_ids=position_ids
    )

    kv_caches = make_groot_language_kv_caches(
        core,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=device,
    )

    ctx_len = torch.full(
        (batch_size,),
        max_seq_len,
        device=device,
        dtype=torch.int32,
    )

    wrapper = GROOTLanguageEngineWrapper(plugin_language).to(device=device).eval()

    sample_inputs = (
        input_embs.to(device=device, dtype=dtype).contiguous(),
        ctx_len.contiguous(),
        *[kv.contiguous() for kv in kv_caches],
    )

    input_names = (
        ["inputs_embeds", "ctx_len"]
        + [f"kv_cache_{i}" for i in range(len(kv_caches))]
    )

    cfg = core.backbone.eagle_model.language_model.config
    head_dim = getattr(
        cfg,
        "head_dim",
        cfg.hidden_size // cfg.num_attention_heads,
    )

    return save_trt_engine_module(
        wrapper,
        sample_inputs,
        engine_dir,
        engine_file="language.engine",
        model_type=model_type,
        component="language",
        input_names=input_names,
        output_names=["context_embs"],
        extra_config={
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "num_layers": len(kv_caches),
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": head_dim,
        },
    )

def save_groot_action_diffusion_engine_for_edge_llm(
    core,
    context_embs,
    state,
    embodiment_id,
    engine_dir,
    *,
    device,
    dtype=torch.float16,
    model_type="groot_action_diffusion",
):
    action_module = GrootStaticDiffusionStep(core.action_head).eval().to(
        device=device,
        dtype=dtype,
    )

    context_embs = context_embs.to(device=device, dtype=dtype).contiguous()
    state = state.to(device=device, dtype=dtype).contiguous()
    embodiment_id = embodiment_id.to(device=device).contiguous()

    sample_inputs = make_compile_inputs(
        action_module,
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
        input_names=[
            "actions",
            "timestep",
            "context_embs",
            "state",
            "embodiment_id",
        ],
        output_names=["pred_velocity"],
        example_output=eager_output,
        extra_config={
            "engine_role": "single_action_denoising_step",
            "action_horizon": int(cfg.action_horizon),
            "action_dim": int(cfg.action_dim),
            "num_inference_timesteps": int(core.action_head.num_inference_timesteps),
            "num_timestep_buckets": int(core.action_head.num_timestep_buckets),
            "context_seq_len": int(context_embs.shape[1]),
            "context_hidden_size": int(context_embs.shape[2]),
            "state_horizon": int(state.shape[1]),
            "state_dim": int(state.shape[2]),
        },
    )

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
    engine_root: str = "/tmp/groot_edge_llm",
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
    register_plugin_op()
    from trt import plugin_converter as _plugin_converter  # noqa: F401,E402
    load_plugin()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    engine_dir = str(pathlib.Path(engine_root) / "visual")
    trt_vision = save_groot_visual_engine_for_edge_llm(
        model,
        pixel_values,
        engine_dir,
        device=device,
        dtype=torch.float16,
        model_type="groot_vision",
    )

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")

    with torch.no_grad():
        eager_image_embs = GROOTVisualFixedInput(
            model,
            pixel_values,
        ).eval().to(device=device, dtype=torch.float16)(pixel_values)

    language_inputs = build_groot_language_inputs(
        model,
        eager_image_embs,
        input_ids,
        attention_mask,
    )

    language_engine_dir = str(pathlib.Path(engine_root) / "language")
    trt_lm = save_groot_lm_engine_for_edge_llm(
        model,
        language_inputs.inputs_embeds,
        language_engine_dir,
        device=device,
        position_ids=None,
        dtype=torch.float16,
        model_type="groot_language",
    )
    
    # -------------------------
    # Action/diffusion engine
    # -------------------------
    print("compiling action diffusion")

    with torch.no_grad():
        context_embs, _, _, _ = build_groot_context_inputs(
            model,
            eager_image_embs,
            input_ids,
            attention_mask,
        )

    action_engine_dir = str(pathlib.Path(engine_root) / "action")
    trt_diffusion = save_groot_action_diffusion_engine_for_edge_llm(
        model,
        context_embs,
        state,
        embodiment_id,
        action_engine_dir,
        device=device,
        dtype=torch.float16,
        model_type="groot_action_diffusion",
    )

    plugin_info = {
        "engine_root": engine_root,
        "vision_engine_dir": str(pathlib.Path(engine_root) / "visual"),
        "language_engine_dir": language_engine_dir,
        "action_engine_dir": action_engine_dir,
        "vision_engine": str(trt_vision),
        "language_engine": str(trt_lm),
        "diffusion_engine": str(trt_diffusion),
        "language_seq_len": int(language_inputs.inputs_embeds.shape[1]),
        "context_seq_len": int(context_embs.shape[1]),
        "context_hidden_size": int(context_embs.shape[2]),
        "state_shape": list(state.shape),
        "embodiment_id": embodiment_id.detach().cpu().tolist(),
    }

    return trt_vision, trt_lm, trt_diffusion, plugin_info


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
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    state, _ = pack_state(
        model_inputs["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )
    state = state.to(device=device, dtype=torch.float16).contiguous()
    embodiment_id = _make_embodiment_id(policy, state, device).contiguous()

    pixel_values = tokenized_data["pixel_values"].to(
        device=device,
        dtype=torch.float16,
    ).contiguous()

    register_plugin_op()
    from trt import plugin_converter as _plugin_converter  # noqa: F401,E402
    load_plugin()

    plugin_settings = {
        **TRT_SETTINGS,
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

    visual = GROOTVisualFixedInput(
        model,
        pixel_values,
    ).eval().to(device=device, dtype=torch.float16)

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
            plugin_settings,
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

    language_inputs = build_groot_language_inputs(
        model,
        trt_image_embs,
        input_ids,
        attention_mask,
    )

    language_max_seq_len = int(language_inputs.inputs_embeds.shape[1])
    if max_seq_len is not None:
        language_max_seq_len = int(max_seq_len)

    plugin_language = make_groot_plugin_language(
        model,
        max_seq_len=language_max_seq_len,
        device=device,
        position_ids=None,
    )

    kv_caches = make_groot_language_kv_caches(
        model,
        batch_size=language_inputs.inputs_embeds.shape[0],
        max_seq_len=language_max_seq_len,
        device=device,
    )

    ctx_len = torch.full(
        (language_inputs.inputs_embeds.shape[0],),
        language_inputs.inputs_embeds.shape[1],
        device=device,
        dtype=torch.int32,
    )

    trt_lm = compile_trt_module(
        plugin_language,
        (language_inputs.inputs_embeds.to(device=device, dtype=torch.float16), kv_caches, ctx_len),
        plugin_settings,
    )

    with torch.no_grad():
        trt_context_embs = run_groot_plugin_language(
            trt_lm,
            model,
            language_inputs.inputs_embeds,
            max_seq_len=language_max_seq_len,
            device=device,
        ).to(device=device, dtype=torch.float16)

    # -------------------------
    # Action/diffusion engine
    # -------------------------
    print("compiling action diffusion")

    action_module = GrootStaticDiffusionStep(model.action_head).eval().to(
        device=device,
        dtype=torch.float16,
    )

    sample_inputs = make_compile_inputs(
        action_module,
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

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    embodiment_id = torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

    start_time = time.perf_counter()

    with torch.autocast("cuda", dtype=torch.float16):
        image_embs = GROOTVisualFixedInput(
            model,
            pixel_values,
        ).eval().to(device=device, dtype=torch.float16)(pixel_values)

        context_embs, _, _, _ = build_groot_context_inputs(
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

        action_module = GrootStaticDiffusionStep(model.action_head).eval().to(
            device=device,
            dtype=torch.float16,
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

    language_inputs = build_groot_language_inputs(
        model,
        image_embs,
        input_ids,
        attention_mask,
    )

    context_embs = run_groot_plugin_language(
        trt_lm,
        model,
        language_inputs.inputs_embeds,
        max_seq_len=int(plugin_info["language_max_seq_len"]),
        device=device,
    ).to(device=device, dtype=torch.float16)

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

def _torch_dtype_from_string(dtype: str) -> torch.dtype:
    if dtype in {"torch.float16", "float16", "fp16"}:
        return torch.float16
    if dtype in {"torch.float32", "float32", "fp32"}:
        return torch.float32
    if dtype in {"torch.int32", "int32"}:
        return torch.int32
    if dtype in {"torch.int64", "int64"}:
        return torch.int64
    if dtype in {"torch.bool", "bool"}:
        return torch.bool
    if dtype in {"torch.uint8", "uint8"}:
        return torch.uint8
    raise ValueError(f"Unsupported tensor dtype in config: {dtype}")


def _dump_tensor_bin(path: pathlib.Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensor = tensor.detach().contiguous().cpu()
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
    path.write_bytes(ctypes.string_at(cpu_tensor.data_ptr(), nbytes))


def _load_tensor_bin(path: pathlib.Path, shape: list[int], dtype: str, device: torch.device) -> torch.Tensor:
    raw = bytearray(path.read_bytes())
    tensor = torch.frombuffer(raw, dtype=_torch_dtype_from_string(dtype)).clone()
    return tensor.reshape(tuple(shape)).to(device=device)

def _load_first_output_tensor(engine_root: pathlib.Path, component: str, path: pathlib.Path, device: torch.device) -> torch.Tensor:
    config = load_engine_config(engine_root, component)
    output = config["outputs"][0]
    return _load_tensor_bin(path, output["shape"], output["dtype"], device)

def _load_named_input_tensor(engine_root: pathlib.Path, component: str, name: str, path: pathlib.Path, device: torch.device) -> torch.Tensor:
    config = load_engine_config(engine_root, component)
    meta = config["inputs"][name]
    return _load_tensor_bin(path, meta["shape"], meta["dtype"], device)

def _run_groot_runtime_stage(runtime_bin: str, engine_root: pathlib.Path, fixture_dir: pathlib.Path, stage: str, plugin_so: str | None) -> None:
    runtime_path = pathlib.Path(runtime_bin)
    if not runtime_path.exists():
        raise FileNotFoundError(
            f"Missing GR00T runtime executable: {runtime_path}. "
            "Build it with: cmake --build gitlab/TensorRT-Edge-LLM/build-plugin-trt10 --target groot_runtime_smoke"
        )

    env = os.environ.copy()
    plugin_path = plugin_so or env.get("EDGELLM_TRT_PLUGIN_SO") or env.get("EDGE_LLM_PLUGIN_SO") or env.get("EDGELLM_PLUGIN_PATH")
    if plugin_path:
        env["EDGELLM_PLUGIN_PATH"] = plugin_path

    subprocess.run(
        [
            str(runtime_path),
            "--engine-root",
            str(engine_root),
            "--fixture-dir",
            str(fixture_dir),
            "--stage",
            stage,
        ],
        check=True,
        env=env,
    )


def _make_embodiment_id(policy, state: torch.Tensor, device: torch.device) -> torch.Tensor:
    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    return torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

@torch.no_grad()
def _print_groot_stage_parity(
    model,
    *,
    pixel_values: torch.Tensor,
    language_inputs: PackedLanguageInputs,
    visual_embeds: torch.Tensor,
    context_embs: torch.Tensor,
    noise: torch.Tensor,
    actions: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    engine_root: pathlib.Path,
    fixture_dir: pathlib.Path,
    runtime_bin: str,
    plugin_so: str | None,
    device: torch.device,
) -> None:
    print("  staged parity:")

    with torch.autocast("cuda", dtype=torch.float16):
        eager_visual_embeds = GROOTVisualFixedInput(
            model,
            pixel_values,
        ).eval().to(device=device, dtype=torch.float16)(pixel_values)
    tensor_error_metrics("    visual_embeds", visual_embeds, eager_visual_embeds)

    eager_context_embs = build_groot_context_from_language_inputs(
        model,
        language_inputs,
    ).to(device=device, dtype=torch.float16)
    tensor_error_metrics("    context_embs", context_embs, eager_context_embs)

    timestep = torch.zeros(
        noise.shape[0],
        device=device,
        dtype=torch.long,
    ).contiguous()
    _dump_tensor_bin(fixture_dir / "timestep.bin", timestep)
    _run_groot_runtime_stage(runtime_bin, engine_root, fixture_dir, "action-step", plugin_so)
    pred_velocity = _load_first_output_tensor(engine_root, "action", fixture_dir / "pred_velocity.bin", device)

    action_module = GrootStaticDiffusionStep(model.action_head).eval().to(
        device=device,
        dtype=torch.float16,
    )
    eager_pred_velocity = action_module(
        noise,
        timestep,
        context_embs,
        state,
        embodiment_id,
    )
    tensor_error_metrics("    action_step_pred_velocity", pred_velocity, eager_pred_velocity)

    eager_actions = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=noise,
            device=device,
            context_embs=context_embs,
            state=state,
            embodiment_id=embodiment_id,
        ),
        GROOTActionAdapter(model.action_head),
    )
    tensor_error_metrics("    action_rollout", actions, eager_actions)

    metrics = compute_action_parity_metrics(actions, eager_actions)
    print(
        f"    action_rollout ADE={metrics['action_ade']:.6f}  "
        f"mean_abs={metrics['mean_abs']:.6f}  max_abs={metrics['max_abs']:.6f}"
    )

@torch.no_grad()
def run_inference_edge_groot(
    model,
    policy,
    create_inputs_fn: Callable[[], dict],
    *,
    plugin_info: dict | None,
    runtime_bin: str,
    seed: int,
    device: torch.device,
    plugin_so: str | None = None,
    stage_parity: bool = True,
) -> tuple[torch.Tensor, dict, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_inputs = create_inputs_fn()
    engine_root = pathlib.Path((plugin_info or {}).get("engine_root", "/tmp/groot_edge_llm"))
    fixture_dir = engine_root / "fixtures" / f"pid_{os.getpid()}"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=torch.float16)

    state, _ = pack_state(
        model_inputs["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )
    state = state.to(device=device, dtype=torch.float16).contiguous()
    embodiment_id = _make_embodiment_id(policy, state, device).contiguous()

    start_time = time.perf_counter()

    _dump_tensor_bin(fixture_dir / "pixel_values.bin", pixel_values)
    _run_groot_runtime_stage(runtime_bin, engine_root, fixture_dir, "visual", plugin_so)
    visual_embeds = _load_first_output_tensor(engine_root, "visual", fixture_dir / "visual_embeds.bin", device)

    language_inputs = build_groot_language_inputs(
        model,
        visual_embeds,
        input_ids,
        attention_mask,
    )
    inputs_embeds = language_inputs.inputs_embeds.to(device=device, dtype=torch.float16).contiguous()

    _dump_tensor_bin(fixture_dir / "inputs_embeds.bin", inputs_embeds)
    _run_groot_runtime_stage(runtime_bin, engine_root, fixture_dir, "language", plugin_so)
    context_embs = _load_first_output_tensor(engine_root, "language", fixture_dir / "context_embs.bin", device)
    context_embs = context_embs.to(device=device, dtype=torch.float16).contiguous()

    noise = torch.randn(
        context_embs.shape[0],
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=context_embs.dtype,
    ).contiguous()

    _dump_tensor_bin(fixture_dir / "initial_actions.bin", noise)
    _dump_tensor_bin(fixture_dir / "context_embs.bin", context_embs)
    _dump_tensor_bin(fixture_dir / "state.bin", state)
    _dump_tensor_bin(fixture_dir / "embodiment_id.bin", embodiment_id)

    _run_groot_runtime_stage(runtime_bin, engine_root, fixture_dir, "action", plugin_so)
    actions = _load_named_input_tensor(engine_root, "action", "actions", fixture_dir / "actions_out.bin", device)

    elapsed = time.perf_counter() - start_time

    if stage_parity:
        _print_groot_stage_parity(
            model,
            pixel_values=pixel_values,
            language_inputs=language_inputs,
            visual_embeds=visual_embeds,
            context_embs=context_embs,
            noise=noise,
            actions=actions,
            state=state,
            embodiment_id=embodiment_id,
            engine_root=engine_root,
            fixture_dir=fixture_dir,
            runtime_bin=runtime_bin,
            plugin_so=plugin_so,
            device=device,
        )

    extra = {
        "fixture_dir": str(fixture_dir),
        "noise": noise,
        "visual_embeds": visual_embeds,
        "context_embs": context_embs,
        "state": state,
        "embodiment_id": embodiment_id,
    }

    return actions, extra, elapsed

def make_groot_create_inputs_fn(processor, data, messages, device):
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

def main() -> int:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.plugin_so:
        os.environ["EDGELLM_TRT_PLUGIN_SO"] = args.plugin_so

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

    create_inputs_fn = make_groot_create_inputs_fn(
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

    if not args.skip_engine or not args.skip_edge:
        if args.skip_export:
            serialized_engine_info = {"engine_root": args.engine_dir}
        else:
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
    edge_times: list[float] = []
    action_ades: list[float] = []
    actionmean_abs: list[float] = []
    engine_action_ades: list[float] = []
    engine_actionmean_abs: list[float] = []
    edge_action_ades: list[float] = []
    edge_actionmean_abs: list[float] = []

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

            if pred_actions_pt is not None:
                trt_metrics = compute_action_parity_metrics(pred_actions_trt, pred_actions_pt)
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

            if pred_actions_pt is not None:
                engine_metrics = compute_action_parity_metrics(pred_actions_engine, pred_actions_pt)
                engine_action_ades.append(engine_metrics["action_ade"])
                engine_actionmean_abs.append(engine_metrics["mean_abs"])
                print(f"  Serialized : {engine_elapsed:7.1f} ms   actionADE={engine_metrics['action_ade']:.6f}  mean_abs={engine_metrics['mean_abs']:.6f}")
            else:
                print(f"  Serialized : {engine_elapsed:7.1f} ms")

        # -- Edge Runtime --------------------------------------------------
        if not args.skip_edge:
            if device.type == "cuda":
                torch.cuda.synchronize(); t = time.perf_counter()
            else:
                t = time.perf_counter()
            pred_actions_edge, extra_edge, _ = run_inference_edge_groot(
                model,
                policy,
                create_inputs_fn,
                plugin_info=edge_plugin_info,
                runtime_bin=args.groot_runtime_bin,
                seed=args.seed,
                device=device,
                plugin_so=args.plugin_so,
                stage_parity=not args.no_stage_parity,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            edge_elapsed = 1000 * (time.perf_counter() - t)
            edge_times.append(edge_elapsed)

            if pred_actions_pt is not None:
                edge_metrics = compute_action_parity_metrics(pred_actions_edge, pred_actions_pt)
                edge_action_ades.append(edge_metrics["action_ade"])
                edge_actionmean_abs.append(edge_metrics["mean_abs"])
                print(f"  Edge Runtime: {edge_elapsed:7.1f} ms   actionADE={edge_metrics['action_ade']:.6f}  mean_abs={edge_metrics['mean_abs']:.6f}")
            else:
                print(f"  Edge Runtime: {edge_elapsed:7.1f} ms")

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations})")
    print("=" * 78)

    if pt_times:
        print_timing("PyTorch GR00T", pt_times[args.warmup:])

    if trt_times:
        print_timing("TRT Plugin FP16", trt_times[args.warmup:])

    if engine_times:
        print_timing("Serialized Engine", engine_times[args.warmup:])

    if edge_times:
        print_timing("Edge Runtime", edge_times[args.warmup:])

    if action_ades:
        print_action_metrics("TRT Action ADE", action_ades[args.warmup:])
        print_action_metrics("TRT Action mean abs", actionmean_abs[args.warmup:])

    if engine_action_ades:
        print_action_metrics("Engine Action ADE", engine_action_ades[args.warmup:])
        print_action_metrics("Engine Action mean abs", engine_actionmean_abs[args.warmup:])

    if edge_action_ades:
        print_action_metrics("Edge Action ADE", edge_action_ades[args.warmup:])
        print_action_metrics("Edge Action mean abs", edge_actionmean_abs[args.warmup:])

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

    if pt_times and edge_times:
        pt_avg = mean(pt_times[args.warmup:])
        edge_avg = mean(edge_times[args.warmup:])
        speedup = pt_avg / edge_avg if edge_avg > 0 else float("nan")
        print(f"  Speedup (Edge vs PyTorch): {speedup:5.2f}x   ({pt_avg:.1f} -> {edge_avg:.1f} ms)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())