from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer

from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION

from trt.action_rollout import ActionRolloutContext, PrefixKVFlowActionAdapter, sample_actions_raw
from trt.compile import compile_trt_module, dump_edge_fixture, save_trt_engine_module
from trt.edge_llm_runtime import run_llm_inference_runtime_smoke
from trt.chat_template import build_pi05_vitrunner_chat_template
from trt.io_spec import (
    PI05_ACTION_ROLLOUT,
    PI05_EDGE_IO,
    PipelineIOSpec,
    action_rollout_extra_config,
)
from trt.data import load_test_data, prepare_policy_batch
from trt.diffusion import StaticActionVelocityStep, PI05PrefixKVStepEncoder
from trt.language import (
    compile_language_trt_with_plugin,
    language_head_dim,
    make_plugin_lm_causal_wrapper,
    pi05_plugin_lm_smoke_check,
    run_prefix_language_eager,
    save_language_engine_for_edge_llm,
    unpack_vla_prefix_language_outputs,
)
from trt.language_builders import build_pi05_language_export_params
from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm
from trt.rope import (
    make_dummy_rope_rotary_cos_sin,
    make_rope_rotary_cos_sin,
)
from trt.measure import (
    compare_action_step,
    compare_language,
    compare_vision,
    compute_action_parity_metrics,
    mean,
    print_action_metrics,
    print_timing,
    tensor_error_metrics,
)
from trt.packing import pack_pi05_prefix
from trt.plugin_utils import (
    patch_vision_attention,
    restore_attention,
    load_plugins_for_trt,
)
from trt.serialize import (
    SerializedModuleSpec,
    SerializedPI05Action,
    SerializedPI05Language,
    SerializedTRTEngine,
    load_serialized_modules,
)
from trt.utils import (
    clone_hf_module_for_export,
    ensure_pi05_paligemma_on_device,
    free_cuda_memory,
    load_policy,
    make_suffix_position_and_mask,
    prepare_policy_inputs,
)
from trt.vision import (
    VisualFixedInput,
    VIT_ENGINE_INPUT_NAME,
    VIT_ENGINE_OUTPUT_NAME,
    nchw_to_hwc,
    run_trt_vision_nchw,
    save_visual_engine_for_edge_llm,
)
from trt.vision_builders import build_pi05_vision_export_params

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
    "use_fp32_acc": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "use_fp32_acc": True,
    "offload_module_to_cpu": True,
}

LANGUAGE_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}

MODEL_ID = "lerobot/pi05_libero"
PALIGEMMA_TOKENIZER_ID = "google/paligemma-3b-pt-224"
SEED = 42
WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LLM_INFERENCE_BIN = (
    WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference"
)
DEFAULT_EDGE_LLM_PLUGIN_SO = (
    WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/libNvInfer_edgellm_plugin.so"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PI0.5 TensorRT engines for TensorRT-Edge-LLM")

    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="PI0.5 policy/model id to load.")
    parser.add_argument("--dataset-id", type=str, default="lerobot/libero", help="LeRobot dataset id used for representative inputs.")
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index used for the compile sample.")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index used for the compile sample.")
    parser.add_argument("--engine-dir", type=str, default="/tmp/pi05_edge_llm", help="Root directory for exported PI0.5 engines.")
    parser.add_argument("--device", type=str, default="cuda", help="Compile device.")
    parser.add_argument(
        "--llm-inference-bin",
        type=str,
        default=str(DEFAULT_LLM_INFERENCE_BIN),
        help="Path to TensorRT-Edge-LLM llm_inference binary for C++ smoke tests.",
    )

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for compile/test tensors.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Static language length override. For PI0.5 this must match the compact prefix length.")
    parser.add_argument("--num-traj-samples", type=int, default=1, help="Compatibility flag; PI0.5 uses one sampled action rollout.")
    parser.add_argument("--max-generation-length", type=int, default=256, help="Compatibility flag; PI0.5 prefix prefill does not generate tokens.")

    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export serialized .engine files and run parity checks; skip in-memory TRT plugin compile.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable extra debug logging/checks.")
    parser.add_argument("--no-accuracy-check", action="store_true", help="Skip eager-vs-TRT accuracy checks.")
    parser.add_argument(
        "--no-stage-parity",
        action="store_true",
        help="Skip staged Edge export vs eager parity diagnostics.",
    )
    parser.add_argument(
        "--run-cpp-smoke",
        action="store_true",
        help="After export, run llm_inference on runtime_smoke/input.json.",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip TensorRT .engine export and load existing engines from --engine-dir.")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip eager PyTorch action rollout.")
    parser.add_argument("--skip-trt", action="store_true", help="Skip Python TRT plugin action rollout.")
    parser.add_argument("--skip-engine", action="store_true", help="Skip Python serialized .engine action rollout.")

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


def get_pi05_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(PALIGEMMA_TOKENIZER_ID)

def _tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu()
    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    return Image.fromarray(img.numpy())


def write_pi05_runtime_smoke_case(
    engine_root: str | pathlib.Path,
    *,
    task_text: str,
    image: torch.Tensor,
    max_generate_length: int = 0,
) -> pathlib.Path:
    """Write llm_inference JSON + PNG under runtime_smoke/."""
    engine_root = pathlib.Path(engine_root)
    smoke_dir = engine_root / "runtime_smoke"
    image_path = smoke_dir / "camera_0.png"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _tensor_image_to_pil(image).save(image_path)

    payload = {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": int(max_generate_length),
        "requests": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path.resolve())},
                            {"type": "text", "text": task_text},
                        ],
                    }
                ],
            }
        ],
    }
    input_path = smoke_dir / "input.json"
    input_path.write_text(json.dumps(payload, indent=2) + "\n")
    return input_path


class PI05VisionEngineAdapter:
    """Serialized VitRunner wrapper: NCHW policy tensors -> HWC engine binding `input`."""

    def __init__(self, engine: SerializedTRTEngine):
        self.engine = engine

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hwc = nchw_to_hwc(pixel_values.to(device=pixel_values.device).contiguous())
        return self.engine({VIT_ENGINE_INPUT_NAME: hwc})[0]


def _run_serialized_pi05_language(
    language_runner: SerializedPI05Language,
    prefix_embs: torch.Tensor,
    *,
    max_seq_len: int,
    device: torch.device,
    position_ids: torch.Tensor | None,
    cfg,
    language_model=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_embs = prefix_embs.to(device=device, dtype=torch.float16).contiguous()
    batch_size = int(prefix_embs.shape[0])
    kv_caches = [
        torch.zeros(
            batch_size,
            2,
            int(cfg.num_key_value_heads),
            int(max_seq_len),
            language_head_dim(cfg),
            device=device,
            dtype=prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]
    ctx_len = torch.full((batch_size,), prefix_embs.shape[1], device=device, dtype=torch.int32)
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        int(max_seq_len),
        device,
        language_model=language_model,
        position_ids=position_ids,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (batch_size, 1),
        int(prefix_embs.shape[1]) - 1,
        device=device,
        dtype=torch.int64,
    )
    return language_runner(
        prefix_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        kv_caches,
    )


@torch.no_grad()
def compare_pi05_edge_pipeline_to_eager(
    core,
    policy,
    *,
    images: list[torch.Tensor],
    img_masks,
    tokens,
    masks,
    trt_image_embs: torch.Tensor | list[torch.Tensor],
    trt_hidden: torch.Tensor,
    trt_prefix_k: torch.Tensor,
    trt_prefix_v: torch.Tensor,
    trt_diffusion: nn.Module,
    device: torch.device,
    seed: int,
) -> None:
    """Stage-by-stage TRT vs eager checks for the PI0.5 Edge-LLM export path."""
    print("\n=== PI0.5 Edge engine parity vs eager ===")

    ensure_pi05_paligemma_on_device(core, device)
    eager_image_embs = [
        core.paligemma_with_expert.embed_image(image)
        for image in images
    ]
    trt_embs = trt_image_embs if isinstance(trt_image_embs, list) else [trt_image_embs]
    for idx, (trt_emb, eager_emb) in enumerate(zip(trt_embs, eager_image_embs)):
        tensor_error_metrics(
            f"vision[{idx}]",
            trt_emb.to(device=device, dtype=torch.float16),
            eager_emb.to(device=device, dtype=torch.float16),
        )

    compact_prefix = pack_pi05_prefix(
        core,
        eager_image_embs,
        img_masks,
        tokens,
        masks,
    )
    eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
        core.paligemma_with_expert.paligemma.model.language_model,
        compact_prefix["inputs_embeds"],
        compact_prefix["attention_mask"],
        compact_prefix["position_ids"],
    )
    tensor_error_metrics(
        "language lm_hidden_states",
        trt_hidden.to(device=device, dtype=torch.float16),
        eager_hidden.to(device=device, dtype=torch.float16),
    )
    tensor_error_metrics(
        "language prefix_k",
        trt_prefix_k.to(device=device, dtype=torch.float16),
        eager_prefix_k.to(device=device, dtype=torch.float16),
    )
    tensor_error_metrics(
        "language prefix_v",
        trt_prefix_v.to(device=device, dtype=torch.float16),
        eager_prefix_v.to(device=device, dtype=torch.float16),
    )

    set_reproducible_seed(seed, device)
    noise = make_pi05_noise(core, tokens.shape[0], device)
    position_ids, attention_mask = make_suffix_position_and_mask(
        core,
        compact_prefix["pad_mask"],
        noise,
        device,
    )
    prefix_k = trt_prefix_k.to(device=device, dtype=noise.dtype).contiguous()
    prefix_v = trt_prefix_v.to(device=device, dtype=noise.dtype).contiguous()
    timestep = torch.full((tokens.shape[0],), 1.0, dtype=torch.float32, device=device)

    eager_action_module = make_static_action_module(core, device)
    with torch.no_grad():
        eager_velocity = eager_action_module(
            noise,
            timestep,
            prefix_k,
            prefix_v,
            position_ids.contiguous(),
            attention_mask.contiguous(),
        )
        trt_velocity = trt_diffusion(
            noise,
            timestep,
            prefix_k,
            prefix_v,
            position_ids.contiguous(),
            attention_mask.contiguous(),
        )
    tensor_error_metrics("diffusion velocity", trt_velocity, eager_velocity)

    eager_actions = sample_actions_raw(
        eager_action_module,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=compact_prefix["pad_mask"],
        ),
        PrefixKVFlowActionAdapter(core, int(core.config.num_inference_steps)),
    )
    trt_actions = sample_actions_raw(
        trt_diffusion,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=compact_prefix["pad_mask"],
        ),
        PrefixKVFlowActionAdapter(core, int(core.config.num_inference_steps)),
    )
    metrics = compute_action_parity_metrics(
        crop_policy_actions(policy, trt_actions),
        crop_policy_actions(policy, eager_actions),
    )
    print(
        f"full rollout action_ade={metrics['action_ade']:.6f}  "
        f"mean_abs={metrics['mean_abs']:.6f}"
    )


def action_output_dim(policy: Any) -> int:
    output_feature = policy.config.output_features.get(ACTION)
    if output_feature is None:
        return int(policy.model.config.max_action_dim)
    return int(output_feature.shape[0])


def crop_policy_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    return actions[..., : action_output_dim(policy)]


@torch.no_grad()
def make_compile_inputs(
    core,
    *,
    batch_size: int,
    prefix_len: int,
    device: torch.device,
):
    chunk_size = core.config.chunk_size
    action_dim = core.config.max_action_dim
    expert_cfg = core.paligemma_with_expert.gemma_expert.model.config
    dtype = next(core.action_in_proj.parameters()).dtype

    x_t = torch.randn(
        batch_size,
        chunk_size,
        action_dim,
        device=device,
        dtype=dtype,
    )

    timestep = torch.ones(
        batch_size,
        device=device,
        dtype=torch.float32,
    )

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

    prefix_pad_masks = torch.ones(
        batch_size,
        prefix_len,
        dtype=torch.bool,
        device=device,
    )

    position_ids, attention_mask = make_suffix_position_and_mask(
        core,
        prefix_pad_masks,
        x_t,
        device,
    )

    return (
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    )

def make_static_action_module(core, device):
    return StaticActionVelocityStep(
        step_encoder=PI05PrefixKVStepEncoder(core),
        action_expert=core.paligemma_with_expert.gemma_expert.model,
        velocity_decoder=core.action_out_proj,
        output_tokens=core.config.chunk_size,
    ).eval().to(device=device)

def make_pi05_noise(core, batch_size: int, device: torch.device) -> torch.Tensor:
    return core.sample_noise(
        (batch_size, core.config.chunk_size, core.config.max_action_dim),
        device,
    )


def validate_language_len(prefix: dict, max_seq_len: int | None) -> int:
    prefix_len = int(prefix["inputs_embeds"].shape[1])
    if max_seq_len is not None and int(max_seq_len) != prefix_len:
        raise ValueError(
            "PI0.5 Edge export uses a compact static prefix. "
            f"--max-seq-len must match compact prefix length {prefix_len}, got {max_seq_len}."
        )
    return prefix_len


def save_lm_engine_for_edge_llm(
    core,
    prefix_embs: torch.Tensor,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    tokenizer,
    position_ids: torch.Tensor | None = None,
    model_type: str = "language",
    io: PipelineIOSpec = PI05_EDGE_IO,
):
    prefix = {"inputs_embeds": prefix_embs}
    if position_ids is not None:
        prefix["position_ids"] = position_ids

    spec = build_pi05_language_export_params(
        core,
        prefix,
        device,
        io=io,
        trt_settings=LANGUAGE_TRT_SETTINGS,
    )
    spec.model_type = model_type

    engine_path = save_language_engine_for_edge_llm(engine_dir, spec)

    save_embedding_table(spec.language_model, engine_dir)
    save_tokenizer_for_edge_llm(
        engine_dir,
        tokenizer=tokenizer,
        chat_template=build_pi05_vitrunner_chat_template(),
    )
    return engine_path

def save_action_diffusion_engine_for_edge_llm(
    core,
    prefix_len: int,
    batch_size: int,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    model_type: str = "action",
    io: PipelineIOSpec = PI05_EDGE_IO,
):
    action_module = make_static_action_module(core, device)
    sample_inputs = make_compile_inputs(
        core,
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
        input_names=list(io.action.input_names),
        output_names=list(io.action.output_names),
        example_output=eager_output,
        extra_config={
            "engine_role": "single_action_denoising_step",
            **action_rollout_extra_config(
                io,
                PI05_ACTION_ROLLOUT,
                num_steps=int(core.config.num_inference_steps),
                chunk_size=int(core.config.chunk_size),
                max_action_dim=int(core.config.max_action_dim),
                prefix_seq_len=int(prefix_len),
                num_layers=int(expert_cfg.num_hidden_layers),
                num_key_value_heads=int(expert_cfg.num_key_value_heads),
                head_dim=int(expert_cfg.head_dim),
            ),
        },
        trt_settings=ACTION_TRT_SETTINGS,
    )

@torch.no_grad()
def _dump_pi05_edge_fixture(
    *,
    engine_root: str | pathlib.Path,
    core,
    policy: Any,
    pixel_values: torch.Tensor,
    compact_prefix: dict,
    lm_hidden_states: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
) -> pathlib.Path:
    set_reproducible_seed(seed, device)
    batch_size = int(compact_prefix["inputs_embeds"].shape[0])
    noise = make_pi05_noise(core, batch_size, device)
    position_ids, attention_mask = make_suffix_position_and_mask(
        core,
        compact_prefix["pad_mask"],
        noise,
        device,
    )

    num_steps = int(core.config.num_inference_steps)
    timestep = torch.full(
        (batch_size,),
        1.0,
        dtype=torch.float32,
        device=device,
    )

    action_module = make_static_action_module(core, device)
    prefix_k = prefix_k.to(device=device, dtype=noise.dtype).contiguous()
    prefix_v = prefix_v.to(device=device, dtype=noise.dtype).contiguous()

    velocity = action_module(
        noise,
        timestep,
        prefix_k,
        prefix_v,
        position_ids.contiguous(),
        attention_mask.contiguous(),
    )
    actions_out = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=compact_prefix["pad_mask"],
        ),
        PrefixKVFlowActionAdapter(core, num_steps),
    )

    velocity_name = io.action.output_names[0]
    return dump_edge_fixture(
        engine_root,
        {
            "pixel_values": pixel_values.to(device=device).contiguous(),
            "inputs_embeds": compact_prefix["inputs_embeds"].to(device=device, dtype=torch.float16),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "lm_hidden_states": lm_hidden_states.to(device=device, dtype=torch.float16),
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "initial_actions": noise,
            "timestep": timestep,
            velocity_name: velocity,
            # Full padded action shape must match the exported action engine x_t tensor.
            "actions_out": actions_out.to(device=device, dtype=noise.dtype),
        },
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
    paligemma = core.paligemma_with_expert.paligemma
    vision_tower = clone_hf_module_for_export(
        paligemma.model.vision_tower,
        device,
        dtype=next(paligemma.model.vision_tower.parameters()).dtype,
    )
    projector = clone_hf_module_for_export(
        paligemma.model.multi_modal_projector,
        device,
        dtype=next(paligemma.model.multi_modal_projector.parameters()).dtype,
        config=paligemma.config,
    )
    images_hwc = nchw_to_hwc(pixel_values)
    visual = VisualFixedInput(
        vision_model=vision_tower,
        projector=projector,
        sample_pixel_values=images_hwc,
        select_layer=-1,
        pixel_shuffle=False,
        force_float32_input=True,
        cast_output_to_input_dtype=True,
    ).eval().to(device=device)
    vision_model = vision_tower.vision_model
    with torch.no_grad():
        siglip_hidden = vision_model.embeddings(pixel_values=pixel_values)
    batch_size = int(siglip_hidden.shape[0])
    seq_len = int(siglip_hidden.shape[1])

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
            (images_hwc,),
            plugin_settings,
        )
    finally:
        if patched:
            restore_attention(patched)
        free_cuda_memory(visual, vision_tower, projector)

    with torch.no_grad():
        trt_image_embs = [
            run_trt_vision_nchw(trt_vision, image.to(device=device))
            for image in images
        ]

    if accuracy_check:
        compare_vision(
            core,
            images,
            lambda image: run_trt_vision_nchw(trt_vision, image.to(device=device)),
        )

    print("compiling PI0.5 language")
    ensure_pi05_paligemma_on_device(core, device)
    with torch.no_grad():
        eager_image_embs = [
            core.paligemma_with_expert.embed_image(image)
            for image in images
        ]
        eager_prefix = pack_pi05_prefix(
            core,
            eager_image_embs,
            img_masks,
            tokens,
            masks,
        )
        eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
            core.paligemma_with_expert.paligemma.model.language_model,
            eager_prefix["inputs_embeds"],
            eager_prefix["attention_mask"],
            eager_prefix["position_ids"],
        )

    trt_prefix = pack_pi05_prefix(
        core,
        trt_image_embs,
        img_masks,
        tokens,
        masks,
        inputs_dtype=torch.float16,
    )
    language_max_seq_len = validate_language_len(trt_prefix, max_seq_len)

    lm = clone_hf_module_for_export(
        core.paligemma_with_expert.paligemma.model.language_model,
        device,
        dtype=torch.float16,
    )
    decoder = getattr(lm, "model", lm)
    cfg = lm.config
    plugin_language = make_plugin_lm_causal_wrapper(
        decoder,
        cfg,
        clone_hf_module_for_export(
            core.paligemma_with_expert.paligemma.lm_head,
            device,
            dtype=torch.float16,
        ),
    )
    trt_lm, trt_max_seq_len = compile_language_trt_with_plugin(
        plugin_language,
        trt_prefix["inputs_embeds"],
        num_layers=int(cfg.num_hidden_layers),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        device=device,
        settings=plugin_settings,
    )
    language_max_seq_len = int(trt_max_seq_len)

    trt_prefix_embs = trt_prefix["inputs_embeds"].to(device=device, dtype=torch.float16)
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
    trt_rope = make_rope_rotary_cos_sin(
        cfg,
        language_max_seq_len,
        device,
        language_model=lm,
        position_ids=trt_prefix["position_ids"],
    )
    free_cuda_memory(lm)
    trt_kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    trt_last_token_ids = torch.full(
        (trt_prefix_embs.shape[0], 1),
        trt_prefix_embs.shape[1] - 1,
        device=device,
        dtype=torch.int64,
    )
    _, trt_hidden, trt_prefix_k, trt_prefix_v = unpack_vla_prefix_language_outputs(
        trt_lm(
            trt_prefix_embs,
            trt_rope,
            trt_ctx_len,
            trt_kvcache_start_index,
            trt_last_token_ids,
            *trt_kv_caches,
        )
    )

    if accuracy_check:
        pi05_plugin_lm_smoke_check(
            core,
            trt_lm,
            trt_prefix["inputs_embeds"],
            max_seq_len=language_max_seq_len,
            device=device,
            attention_mask=trt_prefix["attention_mask"],
            position_ids=trt_prefix["position_ids"],
            prefix_pad_masks=trt_prefix["pad_mask"],
            max_logit_tokens=16,
        )
        compare_language(
            eager_hidden,
            eager_prefix_k,
            eager_prefix_v,
            trt_hidden,
            trt_prefix_k,
            trt_prefix_v,
            trt_prefix["pad_mask"],
        )

    print("compiling PI0.5 action diffusion")
    action_module = make_static_action_module(core, device)

    sample_inputs = make_compile_inputs(
        core,
        batch_size=tokens.shape[0],
        prefix_len=trt_prefix["pad_mask"].shape[1],
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
            trt_prefix["pad_mask"],
            trt_prefix_k,
            trt_prefix_v,
            noise,
            timestep,
            device=device,
        )

    plugin_info = {
        "language_max_seq_len": language_max_seq_len,
        "vision_output_seq_len": int(visual.output_seq_len),
        "prefix_seq_len": int(trt_prefix["pad_mask"].shape[1]),
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
    seed: int = 42,
    max_seq_len: int | None = None,
    engine_root: str | pathlib.Path = "/tmp/pi05_edge_llm",
    io: PipelineIOSpec = PI05_EDGE_IO,
    accuracy_check: bool = True,
    stage_parity: bool = True,
    max_generate_length: int = 0,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict]:
    engine_root = pathlib.Path(engine_root)
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)
    pixel_values = images[0].to(device=device).contiguous()
    # Required by llm_inference (see trt.tokenizer).
    tokenizer = get_pi05_tokenizer()
    lm_cfg = core.paligemma_with_expert.paligemma.model.language_model.config

    ensure_pi05_paligemma_on_device(core, device)
    load_plugins_for_trt()

    print("exporting PI0.5 vision.engine")
    vision_engine_dir = engine_root / "visual"
    vis_params = build_pi05_vision_export_params(
        core,
        pixel_values,
        device,
        io=io,
        trt_settings=VISION_TRT_SETTINGS,
    )
    save_visual_engine_for_edge_llm(
        pixel_values,
        vision_engine_dir,
        vis_params,
        device=device,
    )

    # One entry per camera; each matches image_embed_shape == [B, S, H]
    trt_image_embs = [
        torch.zeros(
            *vis_params.image_embed_shape,
            device=device,
            dtype=vis_params.input_dtype,
        )
        for _ in images
    ]
    with torch.no_grad():
        compact_prefix = pack_pi05_prefix(
            core,
            trt_image_embs,
            img_masks,
            tokens,
            masks,
            inputs_dtype=torch.float16,
        )

    print("exporting PI0.5 language.engine")
    language_max_seq_len = validate_language_len(compact_prefix, max_seq_len)
    language_engine_dir = engine_root / "language"
    language_engine = save_lm_engine_for_edge_llm(
        core,
        compact_prefix["inputs_embeds"],
        language_engine_dir,
        device=device,
        position_ids=compact_prefix["position_ids"],
        io=io,
        tokenizer=tokenizer,
    )
    free_cuda_memory()
    language_runner = SerializedPI05Language(SerializedTRTEngine(language_engine_dir))
    language_model = core.paligemma_with_expert.paligemma.model.language_model
    with torch.no_grad():
        trt_hidden, trt_prefix_k, trt_prefix_v = _run_serialized_pi05_language(
            language_runner,
            compact_prefix["inputs_embeds"],
            max_seq_len=language_max_seq_len,
            device=device,
            position_ids=compact_prefix["position_ids"],
            cfg=lm_cfg,
            language_model=language_model,
        )

    print("exporting PI0.5 diffusion.engine")
    action_engine_dir = engine_root / "action"
    action_engine = save_action_diffusion_engine_for_edge_llm(
        core,
        prefix_len=int(compact_prefix["pad_mask"].shape[1]),
        batch_size=int(tokens.shape[0]),
        engine_dir=action_engine_dir,
        device=device,
        io=io,
    )
    action_runner = SerializedPI05Action(SerializedTRTEngine(action_engine_dir))

    with torch.no_grad():
        eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
            core.paligemma_with_expert.paligemma.model.language_model,
            compact_prefix["inputs_embeds"],
            compact_prefix["attention_mask"],
            compact_prefix["position_ids"],
        )

    if accuracy_check and stage_parity:
        compare_pi05_edge_pipeline_to_eager(
            core,
            policy,
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            trt_image_embs=trt_image_embs,
            trt_hidden=trt_hidden,
            trt_prefix_k=trt_prefix_k,
            trt_prefix_v=trt_prefix_v,
            trt_diffusion=action_runner,
            device=device,
            seed=seed,
        )

    fixture_dir = _dump_pi05_edge_fixture(
        engine_root=engine_root,
        core=core,
        policy=policy,
        pixel_values=pixel_values,
        compact_prefix=compact_prefix,
        lm_hidden_states=eager_hidden,
        prefix_k=eager_prefix_k,
        prefix_v=eager_prefix_v,
        seed=seed,
        device=device,
        io=io,
    )

    task_text = batch.get("task", "")
    if isinstance(task_text, (list, tuple)):
        task_text = task_text[0] if task_text else "pick up the object"
    smoke_input = write_pi05_runtime_smoke_case(
        engine_root,
        task_text=str(task_text),
        image=pixel_values,
        max_generate_length=max_generate_length,
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
        "prefix_seq_len": int(compact_prefix["pad_mask"].shape[1]),
        "chunk_size": int(core.config.chunk_size),
        "max_action_dim": int(core.config.max_action_dim),
        "output_action_dim": action_output_dim(policy),
        "num_inference_steps": int(core.config.num_inference_steps),
        "fixture_dir": str(fixture_dir),
        "runtime_smoke_input": str(smoke_input),
        **io.to_plugin_info(),
    }

    return vision_engine, language_engine, action_engine, plugin_info


@torch.no_grad()
def run_inference_pytorch_pi05(
    core,
    policy,
    batch: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    set_reproducible_seed(seed, device)
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)

    start_time = time.perf_counter()

    image_embs = [
        core.paligemma_with_expert.embed_image(image)
        for image in images
    ]
    prefix = pack_pi05_prefix(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
    )
    _, prefix_k, prefix_v = run_prefix_language_eager(
        core.paligemma_with_expert.paligemma.model.language_model,
        prefix["inputs_embeds"],
        prefix["attention_mask"],
        prefix["position_ids"],
    )

    noise = make_pi05_noise(core, tokens.shape[0], device)
    action_module = make_static_action_module(core, device)
    actions = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=prefix["pad_mask"],
        ),
        PrefixKVFlowActionAdapter(core, core.config.num_inference_steps),
    )

    elapsed = time.perf_counter() - start_time
    actions = crop_policy_actions(policy, actions)

    extra = {
        "noise": noise,
        "image_embs": image_embs,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix["pad_mask"],
    }
    return actions, extra, elapsed


@torch.no_grad()
def run_inference_trt_plugin(
    core,
    policy,
    batch: dict[str, Any],
    *,
    trt_vision,
    trt_lm,
    trt_diffusion,
    plugin_info: dict,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    set_reproducible_seed(seed, device)
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)

    start_time = time.perf_counter()

    image_embs = [
        run_trt_vision_nchw(trt_vision, image.to(device=device))
        for image in images
    ]
    prefix = pack_pi05_prefix(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
        inputs_dtype=torch.float16,
    )

    prefix_embs = prefix["inputs_embeds"].to(device=device, dtype=torch.float16)
    lm = core.paligemma_with_expert.paligemma.model.language_model
    cfg = lm.config
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
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        int(plugin_info["language_max_seq_len"]),
        device,
        language_model=lm,
        position_ids=prefix["position_ids"],
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (prefix_embs.shape[0], 1),
        prefix_embs.shape[1] - 1,
        device=device,
        dtype=torch.int64,
    )
    _, _, prefix_k, prefix_v = unpack_vla_prefix_language_outputs(
        trt_lm(
            prefix_embs,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            last_token_ids,
            *kv_caches,
        )
    )

    noise = make_pi05_noise(core, tokens.shape[0], device)
    actions = sample_actions_raw(
        trt_diffusion,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=prefix["pad_mask"],
        ),
        PrefixKVFlowActionAdapter(core, int(plugin_info["num_inference_steps"])),
    )

    elapsed = time.perf_counter() - start_time
    actions = crop_policy_actions(policy, actions)

    extra = {
        "noise": noise,
        "image_embs": image_embs,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix["pad_mask"],
    }
    return actions, extra, elapsed

def main() -> int:
    args = parse_args()
    configure_torch_runtime()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = load_test_data(
        dataset_id=args.dataset_id,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
    )

    policy = load_policy(PI05Policy, args.model_id, device).to(device).eval()
    core = policy.model.to(device).eval()
    compile_inputs = prepare_policy_batch(
        policy,
        data,
        device,
        args.model_id,
        fill_missing=True,
    )

    print(
        f"model={args.model_id}  dataset={args.dataset_id}  "
        f"episode={args.episode_index}  frame={args.frame_index}  "
        f"iters={args.num_iterations}  warmup={args.warmup}"
    )

    trt_vision = trt_lm = trt_diffusion = plugin_info = None
    serialized_engine_info = None

    if not args.skip_trt and not args.export_only:
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
        ensure_pi05_paligemma_on_device(core, device)
        if trt_vision is not None and not args.skip_export:
            free_cuda_memory(trt_vision, trt_lm, trt_diffusion)
            trt_vision = trt_lm = trt_diffusion = None

        if args.skip_export:
            serialized_engine_info = {"engine_root": args.engine_dir}
        else:
            _, _, _, serialized_engine_info = save_edge_engines_for_edge_llm(
                core,
                policy,
                device,
                compile_inputs,
                seed=args.seed,
                max_seq_len=args.max_seq_len,
                engine_root=args.engine_dir,
                accuracy_check=not args.no_accuracy_check,
                stage_parity=not args.no_stage_parity,
                max_generate_length=0,
            )

    if not args.skip_trt and not args.export_only and trt_vision is None:
        print("recompiling PI0.5 in-memory TRT modules for plugin benchmarking")
        trt_vision, trt_lm, trt_diffusion, plugin_info = compile_trt_with_plugin(
            core,
            policy,
            device,
            compile_inputs,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
            debug=args.debug,
            accuracy_check=False,
        )

    if args.run_cpp_smoke and not args.skip_engine:
        smoke_input = pathlib.Path(serialized_engine_info.get(
            "runtime_smoke_input",
            pathlib.Path(args.engine_dir) / "runtime_smoke" / "input.json",
        ))
        if not smoke_input.exists():
            raise FileNotFoundError(f"Missing runtime smoke input: {smoke_input}")
        print(f"\nRunning C++ llm_inference smoke: {smoke_input}")
        result = run_llm_inference_runtime_smoke(
            engine_root=args.engine_dir,
            input_file=smoke_input,
            llm_inference_bin=args.llm_inference_bin,
            max_generate_length=0,
            dump_output=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=os.sys.stderr)
        if result.returncode != 0:
            print(f"C++ smoke failed with exit code {result.returncode}")
            return result.returncode
        print("C++ smoke completed successfully.")

    if args.export_only:
        return 0

    engine_vision = engine_lm = engine_diffusion = engine_info = None
    if not args.skip_engine:
        engine_vision, engine_lm, engine_diffusion, engine_info = load_serialized_modules(
            serialized_engine_info["engine_root"],
            specs=(
                SerializedModuleSpec("vision", "visual", PI05VisionEngineAdapter),
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
                compile_inputs,
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
                compile_inputs,
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
                compile_inputs,
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
