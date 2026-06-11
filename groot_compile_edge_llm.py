from __future__ import annotations

import os
import argparse
import torch
import copy
import json
import logging

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
    return trt_vision, trt_lm, None, None

def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, messages = load_test_data(
        dataset_id="lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    '''gt_xyz = data["ego_future_xyz"]

    print(
        f"clip_id={args.clip_id}  num_traj_samples={args.num_traj_samples}  "
        f"iters={args.num_iterations}  warmup={args.warmup}  "
        f"skip_pytorch={args.skip_pytorch}  skip_trt={args.skip_trt}"
    )'''

    # create processor and get tokenizer ----
    cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
    processor = get_processor(str(cache_dir), 
        {
            'trust_remote_code': True, 
            'fix_mistral_regex': False
        })
    # ------
    
    model_inputs = prepare_model_inputs(
        processor,
        processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            },
        },
        data,
        messages,
        device,
    )

    # ── load VLA policy ────────────────────────────
    policy = load_policy(GrootPolicy, MODEL_ID, device).to(device).eval()
    # The core model owns the vision, language/context, and action modules used below.
    model = policy._groot_model.to(device).eval()

    # ── Compile TRT engines once (iter -1) ────────────────────────────
    trt_vision = trt_lm = trt_diffusion = plugin_info = None
    #if not args.skip_trt:
    trt_vision, trt_lm, trt_diffusion, plugin_info = compile_trt_with_plugin(
        model,
        policy,
        device,
        model_inputs,
        seed=args.seed,
        max_generation_length=args.max_generation_length,
        num_traj_samples=args.num_traj_samples,
    )

    pt_times: list[float] = []
    pt_ades: list[float] = []
    trt_times: list[float] = []
    trt_ades: list[float] = []
    pt_coc = trt_coc = "(skipped)"

    '''for i in range(args.num_iterations):
        print(f"\n=== iter {i} (clip={args.clip_id}) ===", flush=True)

        # ── PyTorch FP16 baseline ────────────────────────────────────
        if not args.skip_pytorch:
            torch.cuda.synchronize(); t = time.perf_counter()
            pred_xyz_pt, _, extra_pt, _ = run_inference_pytorch(
                model,
                create_inputs_fn,
                seed=args.seed,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )
            torch.cuda.synchronize(); pt_elapsed = 1000 * (time.perf_counter() - t)
            pt_metrics = compute_trajectory_metrics(pred_xyz_pt, gt_xyz)
            pt_coc = str(extra_pt["cot"][0][0, 0])
            pt_times.append(pt_elapsed)
            pt_ades.append(pt_metrics["min_ade"])
            print(f"  PyTorch    : {pt_elapsed:7.1f} ms   minADE={pt_metrics['min_ade']:.4f} m")

        # ── TRT Plugin FP16 ──────────────────────────────────────────
        if not args.skip_trt:
            torch.cuda.synchronize(); t = time.perf_counter()
            pred_xyz_trt, _, extra_trt, _ = run_inference_trt_plugin(
                model,
                create_inputs_fn,
                trt_vision=trt_vision,
                trt_lm=trt_lm,
                trt_diffusion=trt_diffusion,
                plugin_info=plugin_info,
                seed=args.seed,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )
            torch.cuda.synchronize(); trt_elapsed = 1000 * (time.perf_counter() - t)
            trt_metrics = compute_trajectory_metrics(pred_xyz_trt, gt_xyz)
            trt_coc = str(extra_trt["cot"][0][0, 0])
            trt_times.append(trt_elapsed)
            trt_ades.append(trt_metrics["min_ade"])
            print(f"  TRT Plugin : {trt_elapsed:7.1f} ms   minADE={trt_metrics['min_ade']:.4f} m")

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations}, hot iters {args.warmup}–{args.num_iterations - 1})")
    print("=" * 78)
    if pt_times:
        _print_timing("PyTorch FP16",   pt_times[args.warmup :])
        _print_minade("PyTorch FP16",   pt_ades[args.warmup :])
    if trt_times:
        _print_timing("TRT Plugin FP16", trt_times[args.warmup :])
        _print_minade("TRT Plugin FP16", trt_ades[args.warmup :])

    if pt_times and trt_times:
        pt_avg = float(np.mean(pt_times[args.warmup :]))
        trt_avg = float(np.mean(trt_times[args.warmup :]))
        speedup = pt_avg / trt_avg if trt_avg > 0 else float("nan")
        print(f"\n  Speedup (TRT vs PyTorch): {speedup:5.2f}x   ({pt_avg:.1f} → {trt_avg:.1f} ms)")

    print("\nCoC outputs (last iter):")
    print(f"  PyTorch    : {pt_coc[:100]}...")
    print(f"  TRT Plugin : {trt_coc[:100]}...")

    return 0'''

if __name__ == "__main__":
    raise SystemExit(main())