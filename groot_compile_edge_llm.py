from __future__ import annotations

import os
import argparse
import torch
import copy
import json
import logging
import time

from typing import Any

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
    GROOTLanguageEngineWrapper
)
from trt.measure import (
    compare_full_groot_to_eager_actions,
    compare_groot_action_step,
    compare_vision,
    tensor_error_metrics,
)
from trt.plugin_utils import (
    register_plugin_op,
    load_plugin,
    patch_vision_attention,  
    restore_attention,
    infer_siglip_seq_len,
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
    parser.add_argument("--skip-edge", action="store_true", help="Skip Edge/TRT action rollout.")
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

@torch.no_grad()
def compare_groot_context(eager_context_embs, trt_context_embs, attention_mask, name="groot context"):

    compact_eager, _, _, _ = make_groot_context_masks(eager_context_embs, attention_mask)
    compact_trt, _, _, _ = make_groot_context_masks(trt_context_embs, attention_mask)

    tensor_error_metrics(name, compact_trt, compact_eager)

def _select_context_rows(context_embs, row_mask):
    row_mask = row_mask.to(device=context_embs.device, dtype=torch.bool)
    return torch.cat(
        [context_embs[b, row_mask[b], :] for b in range(context_embs.shape[0])],
        dim=0,
    )

@torch.no_grad()
def compare_groot_context_token_types(core, eager_context_embs, trt_context_embs, input_ids, attention_mask, name):
    eagle = core.backbone.eagle_model
    image_token_index = getattr(eagle, "image_token_index", eagle.config.image_token_index)

    valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    image_tokens = (input_ids == image_token_index) & valid
    text_tokens = (input_ids != image_token_index) & valid

    if int(image_tokens.sum().item()) > 0:
        tensor_error_metrics(
            f"{name} image tokens",
            _select_context_rows(trt_context_embs, image_tokens),
            _select_context_rows(eager_context_embs, image_tokens),
        )

    if int(text_tokens.sum().item()) > 0:
        tensor_error_metrics(
            f"{name} text tokens",
            _select_context_rows(trt_context_embs, text_tokens),
            _select_context_rows(eager_context_embs, text_tokens),
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

def compile_trt_with_plugin(
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
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, dict]:
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

    engine_dir = "/tmp/groot_edge_llm/visual"
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

    language_engine_dir = "/tmp/groot_edge_llm/language"
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

    action_engine_dir = "/tmp/groot_edge_llm/action"
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
        "vision_engine_dir": "/tmp/groot_edge_llm/visual",
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

def compute_action_parity_metrics(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> dict[str, float]:
    pred = pred_actions.float()
    target = target_actions.float()

    diff = pred - target
    abs_diff = diff.abs()

    step_l2 = torch.linalg.vector_norm(diff, dim=-1)

    return {
        "action_ade": float(step_l2.mean().item()),
        "action_fde": float(step_l2[..., -1].mean().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
    }

@torch.no_grad()
def run_inference_pytorch_groot(
    model,
    policy,
    model_inputs: dict,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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

def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _print_timing(name: str, times_ms: list[float]) -> None:
    if len(times_ms) == 0:
        return
    print(
        f"  {name:<22} min={min(times_ms):7.1f}  avg={_mean(times_ms):7.1f}  "
        f"max={max(times_ms):7.1f}  std={_std(times_ms):6.1f}  (ms)"
    )


def _print_action_metrics(name: str, values: list[float]) -> None:
    if len(values) == 0:
        return
    print(
        f"  {name:<22} min={min(values):9.6f}  avg={_mean(values):9.6f}  "
        f"max={max(values):9.6f}  std={_std(values):9.6f}"
    )

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

    if not args.skip_export:
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

    pt_times: list[float] = []
    edge_times: list[float] = []
    action_ades: list[float] = []
    action_mean_abs: list[float] = []

    for i in range(args.num_iterations):
        print(f"\n=== iter {i} ===", flush=True)

        eager_actions = None

        if not args.skip_pytorch:
            model_inputs = create_inputs_fn()

            if device.type == "cuda":
                torch.cuda.synchronize()

            eager_actions, eager_extra, pt_elapsed_sec = run_inference_pytorch_groot(
                model,
                policy,
                model_inputs,
                seed=args.seed,
                device=device,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            pt_elapsed_ms = 1000 * pt_elapsed_sec
            pt_times.append(pt_elapsed_ms)

            print(f"  PyTorch GR00T : {pt_elapsed_ms:7.1f} ms")

        if not args.skip_edge:
            print("  Edge GR00T    : skipped until serialized engine runner is wired")

            # Later this block becomes:
            #
            # model_inputs = create_inputs_fn()
            #
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            #
            # edge_actions, edge_extra, edge_elapsed_sec = run_inference_edge_groot(
            #     model,
            #     policy,
            #     model_inputs,
            #     trt_vision=trt_vision,
            #     trt_lm=trt_lm,
            #     trt_diffusion=trt_diffusion,
            #     plugin_info=plugin_info,
            #     seed=args.seed,
            #     device=device,
            # )
            #
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            #
            # edge_elapsed_ms = 1000 * edge_elapsed_sec
            # edge_times.append(edge_elapsed_ms)
            #
            # print(f"  Edge GR00T    : {edge_elapsed_ms:7.1f} ms")
            #
            # if eager_actions is not None:
            #     metrics = compute_action_parity_metrics(edge_actions, eager_actions)
            #     action_ades.append(metrics["action_ade"])
            #     action_mean_abs.append(metrics["mean_abs"])
            #     print(
            #         f"  action ADE={metrics['action_ade']:.6f}  "
            #         f"mean_abs={metrics['mean_abs']:.6f}  "
            #         f"max_abs={metrics['max_abs']:.6f}"
            #     )

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations})")
    print("=" * 78)

    if pt_times:
        _print_timing("PyTorch GR00T", pt_times[args.warmup:])

    if edge_times:
        _print_timing("Edge GR00T", edge_times[args.warmup:])

    if action_ades:
        _print_action_metrics("Action ADE", action_ades[args.warmup:])
        _print_action_metrics("Action mean abs", action_mean_abs[args.warmup:])

    if pt_times and edge_times:
        pt_avg = _mean(pt_times[args.warmup:])
        edge_avg = _mean(edge_times[args.warmup:])
        speedup = pt_avg / edge_avg if edge_avg > 0 else float("nan")
        print(f"\n  Speedup: {speedup:5.2f}x   ({pt_avg:.1f} -> {edge_avg:.1f} ms)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())