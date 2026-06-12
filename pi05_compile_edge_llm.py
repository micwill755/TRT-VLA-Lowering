from __future__ import annotations

import copy
import argparse
import ctypes
import os
import pathlib
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION

from trt.action_rollout import ActionRolloutContext, PI05ActionAdapter, sample_actions_raw
from trt.compile import compile_trt_module, save_trt_engine_module
from trt.data import make_batch
from trt.diffusion import PI05StaticKVDiffusionStep
from trt.language import (
    FlatKVLanguageEngineWrapper,
    compile_language_trt_with_plugin,
    language_head_dim,
    make_plugin_lm_hidden_wrapper,
    pi05_plugin_lm_smoke_check,
    run_prefix_language_eager,
)
from trt.measure import (
    compare_action_step,
    compare_language,
    compare_vision,
    compute_action_parity_metrics,
    mean,
    print_action_metrics,
    print_timing,
)
from trt.packing import PackedLanguageInputs, compact_packed_language_inputs
from trt.plugin_utils import (
    infer_siglip_seq_len,
    load_plugin,
    patch_vision_attention,
    register_plugin_op,
    restore_attention,
    load_plugins_for_trt
)
from trt.serialize import (
    SerializedModuleSpec,
    SerializedPI05Action,
    SerializedPI05Language,
    SerializedPI05Vision,
    load_serialized_modules,
)
from trt.utils import (
    build_packed_prefix_inputs,
    load_policy,
    make_suffix_position_and_mask,
    prepare_policy_inputs,
)
from trt.vision import VisualFixedInput

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}

MODEL_ID = "lerobot/pi05_libero"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PI0.5 TensorRT engines for TensorRT-Edge-LLM")

    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="PI0.5 policy/model id to load.")
    parser.add_argument("--dataset-id", type=str, default="lerobot/libero", help="LeRobot dataset id used for representative inputs.")
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index used for the compile sample.")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index used for the compile sample.")
    parser.add_argument("--engine-dir", type=str, default="/tmp/pi05_edge_llm", help="Root directory for exported PI0.5 engines.")
    parser.add_argument("--plugin-so", type=str, default=os.environ.get("EDGELLM_TRT_PLUGIN_SO") or os.environ.get("EDGE_LLM_PLUGIN_SO"), help="Path to libNvInfer_edgellm_plugin.so.")
    parser.add_argument("--device", type=str, default="cuda", help="Compile device.")

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for compile/test tensors.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Static language length override. For PI0.5 this must match the compact prefix length.")
    parser.add_argument("--num-traj-samples", type=int, default=1, help="Compatibility flag; PI0.5 uses one sampled action rollout.")
    parser.add_argument("--max-generation-length", type=int, default=256, help="Compatibility flag; PI0.5 prefix prefill does not generate tokens.")

    parser.add_argument("--debug", action="store_true", help="Enable extra debug logging/checks.")
    parser.add_argument("--no-accuracy-check", action="store_true", help="Skip eager-vs-TRT accuracy checks.")
    parser.add_argument("--skip-export", action="store_true", help="Skip TensorRT .engine export and load existing engines from --engine-dir.")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip eager PyTorch action rollout.")
    parser.add_argument("--skip-trt", action="store_true", help="Skip Python TRT plugin action rollout.")
    parser.add_argument("--skip-engine", action="store_true", help="Skip Python serialized .engine action rollout.")

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument("--run-edge", dest="skip_edge", action="store_false", help="Attempt the PI0.5 C++ Edge runtime smoke path.")
    edge_group.add_argument("--skip-edge", dest="skip_edge", action="store_true", help="Skip the C++ Edge runtime smoke path.")
    parser.set_defaults(skip_edge=True)

    parser.add_argument("--num-iterations", type=int, default=12, help="Total timing iterations including warmup.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations to exclude from summary.")

    return parser.parse_args()


def set_reproducible_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_pi05_batch(policy: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return make_batch(
        policy,
        args.model_id,
        device,
        fill_missing=True,
        dataset_id=args.dataset_id,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
    )


def make_pi05_create_inputs_fn(batch: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    def create_inputs() -> dict[str, Any]:
        return batch

    return create_inputs


def action_output_dim(policy: Any) -> int:
    output_feature = policy.config.output_features.get(ACTION)
    if output_feature is None:
        return int(policy.model.config.max_action_dim)
    return int(output_feature.shape[0])


def crop_policy_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    return actions[..., : action_output_dim(policy)]


def make_pi05_action_compile_inputs(core, action_step, batch_size: int, prefix_len: int, device: torch.device):
    chunk_size = core.config.chunk_size
    action_dim = core.config.max_action_dim
    expert_cfg = core.paligemma_with_expert.gemma_expert.model.config
    dtype = next(action_step.parameters()).dtype

    x_t = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=dtype)
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    prefix_k = torch.zeros(
        expert_cfg.num_hidden_layers,
        batch_size,
        expert_cfg.num_key_value_heads,
        prefix_len,
        expert_cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    prefix_v = torch.zeros_like(prefix_k)
    prefix_pad_masks = torch.ones(batch_size, prefix_len, dtype=torch.bool, device=device)

    position_ids, attention_mask = make_suffix_position_and_mask(core, prefix_pad_masks, x_t, device)
    return x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask


def make_pi05_noise(core, batch_size: int, device: torch.device) -> torch.Tensor:
    return core.sample_noise(
        (batch_size, core.config.chunk_size, core.config.max_action_dim),
        device,
    )


@torch.no_grad()
def prepare_compact_prefix(
    core,
    image_embs: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    *,
    inputs_dtype: torch.dtype | None = None,
) -> PackedLanguageInputs:
    prefix = build_packed_prefix_inputs(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
    )
    if inputs_dtype is not None:
        prefix = prefix.with_inputs_embeds(prefix.inputs_embeds.to(inputs_dtype))
    return compact_packed_language_inputs(prefix)


def validate_language_len(compact_prefix: PackedLanguageInputs, max_seq_len: int | None) -> int:
    prefix_len = int(compact_prefix.inputs_embeds.shape[1])
    if max_seq_len is not None and int(max_seq_len) != prefix_len:
        raise ValueError(
            "PI0.5 Edge export uses a compact static prefix. "
            f"--max-seq-len must match compact prefix length {prefix_len}, got {max_seq_len}."
        )
    return prefix_len


def save_pi05_visual_engine_for_edge_llm(
    core,
    pixel_values: torch.Tensor,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    model_type: str = "pi05_vision",
):
    pixel_values = pixel_values.to(device=device).contiguous()
    visual = VisualFixedInput(
        vision_model=core.paligemma_with_expert.paligemma.model.vision_tower,
        projector=core.paligemma_with_expert.paligemma.model.multi_modal_projector,
        sample_pixel_values=pixel_values,
        select_layer=-1,
        pixel_shuffle=False,
        force_float32_input=True,
        cast_output_to_input_dtype=True,
    )
    vision_model = core.paligemma_with_expert.paligemma.model.vision_tower.vision_model

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
            output_names=["image_embeds"],
            example_output=eager_output,
            extra_config={
                "siglip_batch_size": batch_size,
                "siglip_seq_len": seq_len,
                "num_image_tokens": int(eager_output.shape[1]),
                "hidden_size": int(eager_output.shape[2]),
            },
        )
    finally:
        if patched:
            restore_attention(patched)


def save_pi05_lm_engine_for_edge_llm(
    core,
    prefix_embs: torch.Tensor,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    position_ids: torch.Tensor | None,
    model_type: str = "pi05_language",
):
    prefix_embs = prefix_embs.to(device=device, dtype=torch.float16).contiguous()
    max_seq_len = int(prefix_embs.shape[1])
    batch_size = int(prefix_embs.shape[0])

    lm = copy.deepcopy(
        core.paligemma_with_expert.paligemma.model.language_model
    ).to(device=device, dtype=torch.float16).eval()
    decoder = getattr(lm, "model", lm)
    cfg = lm.config

    lm_wrapper = make_plugin_lm_hidden_wrapper(
        decoder,
        cfg,
        max_seq_len=max_seq_len,
        device=device,
        position_ids=position_ids,
        return_prefix_kv=True,
    )

    kv_caches = [
        torch.zeros(
            batch_size,
            2,  # key + value
            int(cfg.num_key_value_heads),
            max_seq_len,
            language_head_dim(cfg),
            device=device,
            dtype=prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]

    ctx_len = torch.full(
        (batch_size,),
        max_seq_len,
        device=device,
        dtype=torch.int32,
    )

    wrapper = FlatKVLanguageEngineWrapper(lm_wrapper).to(device=device).eval()
    sample_inputs = (
        prefix_embs,
        ctx_len.contiguous(),
        *[kv.contiguous() for kv in kv_caches],
    )

    input_names = (
        ["inputs_embeds", "ctx_len"]
        + [f"kv_cache_{i}" for i in range(len(kv_caches))]
    )

    with torch.no_grad():
        example_output = wrapper(*sample_inputs)

    return save_trt_engine_module(
        wrapper,
        sample_inputs,
        engine_dir,
        engine_file="language.engine",
        model_type=model_type,
        component="language",
        input_names=input_names,
        output_names=["hidden_states", "prefix_k", "prefix_v"],
        example_output=example_output,
        extra_config={
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "num_layers": int(cfg.num_hidden_layers),
            "hidden_size": int(cfg.hidden_size),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "head_dim": int(cfg.head_dim),
        },
    )


def save_pi05_action_diffusion_engine_for_edge_llm(
    core,
    prefix_len: int,
    batch_size: int,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    model_type: str = "pi05_action_diffusion",
):
    action_module = PI05StaticKVDiffusionStep(core).eval().to(device=device)
    sample_inputs = make_pi05_action_compile_inputs(
        core,
        action_module,
        batch_size=batch_size,
        prefix_len=prefix_len,
        device=device,
    )
    sample_inputs = tuple(
        x.contiguous() if isinstance(x, torch.Tensor) else x
        for x in sample_inputs
    )

    with torch.no_grad():
        eager_output = action_module(*sample_inputs)

    expert_cfg = core.paligemma_with_expert.gemma_expert.model.config

    return save_trt_engine_module(
        action_module,
        sample_inputs,
        engine_dir,
        engine_file="diffusion.engine",
        model_type=model_type,
        component="diffusion",
        input_names=[
            "x_t",
            "timestep",
            "prefix_k",
            "prefix_v",
            "position_ids",
            "attention_mask",
        ],
        output_names=["velocity"],
        example_output=eager_output,
        extra_config={
            "engine_role": "single_action_denoising_step",
            "num_inference_steps": int(core.config.num_inference_steps),
            "chunk_size": int(core.config.chunk_size),
            "max_action_dim": int(core.config.max_action_dim),
            "prefix_seq_len": int(prefix_len),
            "num_layers": int(expert_cfg.num_hidden_layers),
            "num_key_value_heads": int(expert_cfg.num_key_value_heads),
            "head_dim": int(expert_cfg.head_dim),
        },
        trt_settings=ACTION_TRT_SETTINGS,
    )

def compile_trt_with_plugin(
    core: nn.Module,
    policy: Any,
    device: torch.device,
    batch: dict[str, Any],
    *,
    seed: int,
    max_seq_len: int | None = None,
    debug: bool = False,
    accuracy_check: bool = True
) -> tuple[nn.Module, nn.Module, nn.Module, dict]:
    del debug
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)
    pixel_values = images[0].to(device=device).contiguous()

    load_plugins_for_trt()

    plugin_settings = {
        **TRT_SETTINGS,
        "use_python_runtime": True,
    }
    action_settings = {
        **ACTION_TRT_SETTINGS,
        "use_python_runtime": True,
    }

    print("compiling PI0.5 vision")
    visual = PI05VisualEmbed(core).eval().to(device=device)
    vision_model = core.paligemma_with_expert.paligemma.model.vision_tower.vision_model
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
        trt_image_embs = [trt_vision(image.to(device=device).contiguous()) for image in images]

    if accuracy_check:
        compare_vision(core, images, trt_vision)

    print("compiling PI0.5 language")
    with torch.no_grad():
        eager_image_embs = [
            core.paligemma_with_expert.embed_image(image)
            for image in images
        ]
        eager_prefix = prepare_compact_prefix(
            core,
            eager_image_embs,
            img_masks,
            tokens,
            masks,
        )
        eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
            core.paligemma_with_expert.paligemma.model.language_model,
            eager_prefix.inputs_embeds,
            eager_prefix.attention_mask,
            eager_prefix.position_ids,
        )

    trt_prefix = prepare_compact_prefix(
        core,
        trt_image_embs,
        img_masks,
        tokens,
        masks,
        inputs_dtype=torch.float16,
    )
    language_max_seq_len = validate_language_len(trt_prefix, max_seq_len)

    lm = copy.deepcopy(
        core.paligemma_with_expert.paligemma.model.language_model
    ).to(device=device, dtype=torch.float16).eval()
    decoder = getattr(lm, "model", lm)
    cfg = lm.config
    plugin_language = make_plugin_lm_hidden_wrapper(
        decoder,
        cfg,
        max_seq_len=language_max_seq_len,
        device=device,
        position_ids=trt_prefix.position_ids,
        return_prefix_kv=True,
    )
    trt_lm, trt_max_seq_len = compile_language_trt_with_plugin(
        plugin_language,
        trt_prefix.inputs_embeds,
        num_layers=int(cfg.num_hidden_layers),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        device=device,
        settings=plugin_settings,
    )
    language_max_seq_len = int(trt_max_seq_len)

    trt_prefix_embs = trt_prefix.inputs_embeds.to(device=device, dtype=torch.float16)
    trt_kv_caches = [
        torch.zeros(
            int(trt_prefix_embs.shape[0]),
            2,  # key + value
            int(cfg.num_key_value_heads),
            language_max_seq_len,
            language_head_dim(cfg),
            device=device,
            dtype=trt_prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]
    trt_ctx_len = torch.full(
        (trt_prefix_embs.shape[0],),
        trt_prefix_embs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    trt_hidden, trt_prefix_k, trt_prefix_v = trt_lm(
        trt_prefix_embs,
        trt_kv_caches,
        trt_ctx_len,
    )

    if accuracy_check:
        pi05_plugin_lm_smoke_check(
            core,
            trt_lm,
            trt_prefix.inputs_embeds,
            max_seq_len=language_max_seq_len,
            device=device,
            attention_mask=trt_prefix.attention_mask,
            position_ids=trt_prefix.position_ids,
            prefix_pad_masks=trt_prefix.pad_mask,
            max_logit_tokens=16,
        )
        compare_language(
            eager_hidden,
            eager_prefix_k,
            eager_prefix_v,
            trt_hidden,
            trt_prefix_k,
            trt_prefix_v,
            trt_prefix.pad_mask,
        )

    print("compiling PI0.5 action diffusion")
    action_module = PI05StaticKVDiffusionStep(core).eval().to(device=device)
    sample_inputs = make_pi05_action_compile_inputs(
        core,
        action_module,
        batch_size=tokens.shape[0],
        prefix_len=trt_prefix.pad_mask.shape[1],
        device=device,
    )
    trt_diffusion = compile_trt_module(
        action_module,
        sample_inputs,
        action_settings,
    )

    if accuracy_check:
        set_reproducible_seed(seed, device)
        noise = make_pi05_noise(core, tokens.shape[0], device)
        timestep = torch.ones(tokens.shape[0], dtype=torch.float32, device=device)
        compare_action_step(
            core,
            action_module,
            trt_diffusion,
            trt_prefix.pad_mask,
            trt_prefix_k,
            trt_prefix_v,
            noise,
            timestep,
            device=device,
        )

    plugin_info = {
        "language_max_seq_len": language_max_seq_len,
        "prefix_seq_len": int(trt_prefix.pad_mask.shape[1]),
        "prefix_k_shape": list(trt_prefix_k.shape),
        "prefix_v_shape": list(trt_prefix_v.shape),
        "chunk_size": int(core.config.chunk_size),
        "max_action_dim": int(core.config.max_action_dim),
        "output_action_dim": action_output_dim(policy),
        "num_inference_steps": int(core.config.num_inference_steps),
    }

    return trt_vision, trt_lm, trt_diffusion, plugin_info


def save_edge_engines_for_edge_llm(
    core: nn.Module,
    policy: Any,
    device: torch.device,
    batch: dict[str, Any],
    *,
    max_seq_len: int | None = None,
    engine_root: str | pathlib.Path = "/tmp/pi05_edge_llm"
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict]:
    engine_root = pathlib.Path(engine_root)
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)
    pixel_values = images[0].to(device=device).contiguous()

    load_plugins_for_trt()

    print("exporting PI0.5 vision.engine")
    vision_engine_dir = engine_root / "visual"
    vision_engine = save_pi05_visual_engine_for_edge_llm(
        core,
        pixel_values,
        vision_engine_dir,
        device=device,
    )

    print("exporting PI0.5 language.engine")
    with torch.no_grad():
        eager_image_embs = [
            core.paligemma_with_expert.embed_image(image)
            for image in images
        ]
        compact_prefix = prepare_compact_prefix(
            core,
            eager_image_embs,
            img_masks,
            tokens,
            masks,
            inputs_dtype=torch.float16,
        )

    language_max_seq_len = validate_language_len(compact_prefix, max_seq_len)
    language_engine_dir = engine_root / "language"
    language_engine = save_pi05_lm_engine_for_edge_llm(
        core,
        compact_prefix.inputs_embeds,
        language_engine_dir,
        device=device,
        position_ids=compact_prefix.position_ids,
    )

    print("exporting PI0.5 diffusion.engine")
    action_engine_dir = engine_root / "action"
    action_engine = save_pi05_action_diffusion_engine_for_edge_llm(
        core,
        prefix_len=int(compact_prefix.pad_mask.shape[1]),
        batch_size=int(tokens.shape[0]),
        engine_dir=action_engine_dir,
        device=device,
    )

    plugin_info = {
        "engine_root": str(engine_root),
        "vision_engine_dir": str(vision_engine_dir),
        "language_engine_dir": str(language_engine_dir),
        "action_engine_dir": str(action_engine_dir),
        "vision_engine": str(vision_engine),
        "language_engine": str(language_engine),
        "diffusion_engine": str(action_engine),
        "language_max_seq_len": language_max_seq_len,
        "prefix_seq_len": int(compact_prefix.pad_mask.shape[1]),
        "chunk_size": int(core.config.chunk_size),
        "max_action_dim": int(core.config.max_action_dim),
        "output_action_dim": action_output_dim(policy),
        "num_inference_steps": int(core.config.num_inference_steps),
    }

    return vision_engine, language_engine, action_engine, plugin_info


@torch.no_grad()
def run_inference_pytorch_pi05(
    core,
    policy,
    create_inputs_fn: Callable[[], dict[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    set_reproducible_seed(seed, device)
    batch = create_inputs_fn()
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)

    start_time = time.perf_counter()

    image_embs = [
        core.paligemma_with_expert.embed_image(image)
        for image in images
    ]
    prefix = prepare_compact_prefix(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
    )
    _, prefix_k, prefix_v = run_prefix_language_eager(
        core.paligemma_with_expert.paligemma.model.language_model,
        prefix.inputs_embeds,
        prefix.attention_mask,
        prefix.position_ids,
    )

    noise = make_pi05_noise(core, tokens.shape[0], device)
    action_module = PI05StaticKVDiffusionStep(core).eval().to(device=device)
    actions = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=prefix.pad_mask,
        ),
        PI05ActionAdapter(core, core.config.num_inference_steps),
    )

    elapsed = time.perf_counter() - start_time
    actions = crop_policy_actions(policy, actions)

    extra = {
        "noise": noise,
        "image_embs": image_embs,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix.pad_mask,
    }
    return actions, extra, elapsed


@torch.no_grad()
def run_inference_trt_plugin(
    core,
    policy,
    create_inputs_fn: Callable[[], dict[str, Any]],
    *,
    trt_vision,
    trt_lm,
    trt_diffusion,
    plugin_info: dict,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    set_reproducible_seed(seed, device)
    batch = create_inputs_fn()
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)

    start_time = time.perf_counter()

    image_embs = [
        trt_vision(image.to(device=device).contiguous())
        for image in images
    ]
    prefix = prepare_compact_prefix(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
        inputs_dtype=torch.float16,
    )

    prefix_embs = prefix.inputs_embeds.to(device=device, dtype=torch.float16)
    cfg = core.paligemma_with_expert.paligemma.model.language_model.config
    kv_caches = [
        torch.zeros(
            int(prefix_embs.shape[0]),
            2,  # key + value
            int(cfg.num_key_value_heads),
            int(plugin_info["language_max_seq_len"]),
            language_head_dim(cfg),
            device=device,
            dtype=prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]
    ctx_len = torch.full(
        (prefix_embs.shape[0],),
        prefix_embs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    _, prefix_k, prefix_v = trt_lm(prefix_embs, kv_caches, ctx_len)

    noise = make_pi05_noise(core, tokens.shape[0], device)
    actions = sample_actions_raw(
        trt_diffusion,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=prefix.pad_mask,
        ),
        PI05ActionAdapter(core, int(plugin_info["num_inference_steps"])),
    )

    elapsed = time.perf_counter() - start_time
    actions = crop_policy_actions(policy, actions)

    extra = {
        "noise": noise,
        "image_embs": image_embs,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix.pad_mask,
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

def run_inference_edge_pi05(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "PI0.5 C++ Edge runtime smoke is not wired in this checkout yet. "
        "The exported engines use the PI0.5 prefix-KV action contract "
        "(x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask), "
        "while the current C++ smoke runner is GR00T-specific."
    )

def main() -> int:
    args = parse_args()
    configure_torch_runtime()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    policy = load_policy(PI05Policy, args.model_id, device).to(device).eval()
    core = policy.model.to(device).eval()
    batch = load_pi05_batch(policy, args, device)
    create_inputs_fn = make_pi05_create_inputs_fn(batch)
    compile_inputs = create_inputs_fn()

    print(
        f"model={args.model_id}  dataset={args.dataset_id}  "
        f"episode={args.episode_index}  frame={args.frame_index}  "
        f"iters={args.num_iterations}  warmup={args.warmup}"
    )

    trt_vision = trt_lm = trt_diffusion = plugin_info = None
    serialized_engine_info = None

    if not args.skip_trt:
        trt_vision, trt_lm, trt_diffusion, plugin_info = compile_trt_with_plugin(
            core,
            policy,
            device,
            compile_inputs,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
            debug=args.debug,
            accuracy_check=not args.no_accuracy_check
        )

    if not args.skip_engine:
        if args.skip_export:
            serialized_engine_info = {"engine_root": args.engine_dir}
        else:
            _, _, _, serialized_engine_info = save_edge_engines_for_edge_llm(
                core,
                policy,
                device,
                compile_inputs,
                max_seq_len=args.max_seq_len,
                engine_root=args.engine_dir
            )

    if not args.skip_edge:
        run_inference_edge_pi05()

    engine_vision = engine_lm = engine_diffusion = engine_info = None
    if not args.skip_engine:
        engine_vision, engine_lm, engine_diffusion, engine_info = load_serialized_modules(
            serialized_engine_info["engine_root"],
            specs=(
                SerializedModuleSpec("vision", "visual", SerializedPI05Vision),
                SerializedModuleSpec("language", "language", SerializedPI05Language),
                SerializedModuleSpec("action", "action", SerializedPI05Action),
            ),
            plugin_info_aliases={
                "language_max_seq_len": ("language", "max_seq_len"),
                "prefix_seq_len": ("action", "prefix_seq_len"),
                "chunk_size": ("action", "chunk_size"),
                "max_action_dim": ("action", "max_action_dim"),
                "num_inference_steps": ("action", "num_inference_steps"),
            },
        )
        engine_info.setdefault("output_action_dim", action_output_dim(policy))

    pt_times: list[float] = []
    trt_times: list[float] = []
    engine_times: list[float] = []
    action_ades: list[float] = []
    actionmean_abs: list[float] = []
    engine_action_ades: list[float] = []
    engine_actionmean_abs: list[float] = []

    for i in range(args.num_iterations):
        print(f"\n=== iter {i} ===", flush=True)

        pred_actions_pt = None

        if not args.skip_pytorch:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            pred_actions_pt, _, _ = run_inference_pytorch_pi05(
                core,
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

        if not args.skip_trt:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            pred_actions_trt, _, _ = run_inference_trt_plugin(
                core,
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

        if not args.skip_engine:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            pred_actions_engine, _, _ = run_inference_trt_plugin(
                core,
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

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations})")
    print("=" * 78)

    if pt_times:
        print_timing("PyTorch PI0.5", pt_times[args.warmup:])

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
