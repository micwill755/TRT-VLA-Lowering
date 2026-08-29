from __future__ import annotations

import os
import sys
import logging
import torch_tensorrt
import torch
import argparse
import logging
import copy
import time

from pathlib import Path

if hasattr(getattr(torch_tensorrt, "logging", None), "set_level"):
    torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
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
from trt.data import create_pil_messages, prepare_model_inputs
from trt.utils import (
    configure_thor_pytorch,
    force_hf_attention,
    free_cuda_memory,
    move_pi05_diffusion_modules_to_device,
    release_serialized_trt_engine,
)

configure_thor_pytorch()
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
from trt.compile import make_input_spec, save_trt_engine_module
from trt.compile_stage_timing import print_stage_breakdown, stage_timing
from trt.executor.models.pi05.load.serialize import SerializedPi05Language
from trt.io_spec import VLA_LANGUAGE_INPUT_NAMES, VLA_LANGUAGE_OUTPUT_NAMES
from trt.language import language_edge_llm_config, language_edge_trt_settings, make_language_edge_input_specs
from trt.serialize import SerializedTRTEngine

from typing import Any

from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
from torchao.quantization.granularity import PerRow
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

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
    # Thor: offload_module_to_cpu balloons host RSS during TRT build (~38GB OOM kill).
    "offload_module_to_cpu": False,
    "use_explicit_typing": False,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    #"offload_module_to_cpu": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
}


def _is_fp8_weight(weight: torch.Tensor) -> bool:
    return hasattr(weight, "qdata")


_FP8_DQACT_CONFIG = Float8DynamicActivationFloat8WeightConfig(
    granularity=PerRow(),
    # AUTO picks MSLK/CUTLASS on this box and fails with "cutlass cannot initialize".
    kernel_preference=KernelPreference.TORCH,
    set_inductor_config=False,
)


def quantize_fp8_dqact(module: torch.nn.Module, name: str) -> None:
    """FP8 weights + dynamic FP8 activations so Linear can use scaled_mm (FP8xFP8)."""
    n_before = sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))
    # PerRow float8 requires bf16 high-precision weights.
    module.to(dtype=torch.bfloat16)
    quantize_(module, _FP8_DQACT_CONFIG)
    n_quant = sum(
        1
        for m in module.modules()
        if isinstance(m, torch.nn.Linear) and _is_fp8_weight(m.weight)
    )
    print(f"[torchao fp8 dqact] {name}: quantized {n_quant}/{n_before} Linear layers")


def tensor_storage_nbytes(t: torch.Tensor) -> tuple[int, str]:
    """Count backing storage. Float8Tensor.dtype is the original hp dtype."""
    if _is_fp8_weight(t):
        n = t.qdata.numel() * t.qdata.element_size()
        scale = getattr(t, "scale", None)
        if scale is not None:
            n += scale.numel() * scale.element_size()
        return n, str(t.qdata.dtype).replace("torch.", "")
    return t.numel() * t.element_size(), str(t.dtype).replace("torch.", "")


def _mib(n_bytes: int | float) -> float:
    return float(n_bytes) / (1024 ** 2)


def param_nbytes_by_dtype(module: torch.nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in module.parameters():
        n, key = tensor_storage_nbytes(p)
        out[key] = out.get(key, 0) + n
    return out


def total_param_nbytes(module: torch.nn.Module) -> int:
    return sum(tensor_storage_nbytes(p)[0] for p in module.parameters())


def format_dtype_mib(by_dtype: dict[str, int]) -> str:
    parts = [
        f"{k}={_mib(v):.1f}MiB"
        for k, v in sorted(by_dtype.items(), key=lambda kv: -kv[1])
    ]
    return ", ".join(parts) if parts else "empty"


def cuda_mem_mib() -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    return (
        _mib(torch.cuda.memory_allocated()),
        _mib(torch.cuda.memory_reserved()),
        _mib(torch.cuda.max_memory_allocated()),
    )


def print_mem(tag: str, module: torch.nn.Module | None = None) -> None:
    alloc, reserved, peak = cuda_mem_mib()
    extra = ""
    if module is not None:
        extra = (
            f"  params={_mib(total_param_nbytes(module)):.1f}MiB "
            f"({format_dtype_mib(param_nbytes_by_dtype(module))})"
        )
    print(
        f"[mem] {tag}: cuda_alloc={alloc:.1f}MiB  "
        f"reserved={reserved:.1f}MiB  peak={peak:.1f}MiB{extra}"
    )


def print_module_param_mem(tag: str, parts: dict[str, torch.nn.Module]) -> None:
    print(f"[mem] {tag} param sizes:")
    for name, mod in parts.items():
        print(
            f"        {name:24s} {_mib(total_param_nbytes(mod)):8.1f} MiB  "
            f"({format_dtype_mib(param_nbytes_by_dtype(mod))})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply torchao FP8 dynamic-activation + FP8-weight quantization (default: on).",
    )
    parser.add_argument(
        "--skip-trt",
        action="store_true",
        help="Eager-only: skip TensorRT export/compile (for quant vs fp16 metrics).",
    )
    return parser.parse_args()


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
    # PI05Policy.__init__ moves the model to config.device in fp32. On Thor the
    # default 6–12 GB GPU carveout cannot hold ~16 GB fp32 weights; init on CPU
    # and let main() cast to fp16 before the first GPU transfer.
    config = PI05Config(
        device="cpu",
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
    policy = PI05Policy(config).eval()
    return config, policy

def main():
    args = parse_args()
    quantize = bool(args.quantize)
    skip_trt = bool(args.skip_trt)
    print(f"quantize={quantize}  skip_trt={skip_trt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if not skip_trt:
        load_plugins_for_trt()
    
    dtype = torch.float16
    # PerRow FP8 dynamic-act needs bf16 as the high-precision compute dtype.
    quant_compute_dtype = torch.bfloat16 if quantize else dtype

    # step 1 - load policy, retrieve vision, lm diffusion, 
    # create processors, data sample and replace attention
    config, policy = load_config(device)
    model = policy.model.to(device=device, dtype=dtype).eval()
    paligemma = model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    language = paligemma.language_model
    select_layer = -1
    lm_head = model.paligemma_with_expert.paligemma.lm_head

    def _param_parts() -> dict[str, torch.nn.Module]:
        return {
            "vision": vision,
            "language": language,
            "lm_head": lm_head,
            "gemma_expert": model.paligemma_with_expert.gemma_expert,
            "action_in_proj": model.action_in_proj,
            "action_out_proj": model.action_out_proj,
            "time_mlp_in": model.time_mlp_in,
            "time_mlp_out": model.time_mlp_out,
            "full_model": model,
        }

    print_module_param_mem("after fp16 load", _param_parts())
    print_mem("after fp16 load", model)

    # FP8 weight-only: language linears only. Vision stays fp32 (SigLIP).
    # Gemma expert is quantized later — move_pi05_diffusion_modules_to_device
    # recasts with .to(dtype=fp16) and would wipe Float8Tensor weights.
    if quantize:
        quantize_fp8_dqact(language, "language")
        quantize_fp8_dqact(lm_head, "lm_head")
        print_module_param_mem("after language fp8 dqact", _param_parts())
        print_mem("after language fp8 dqact", model)
    else:
        print("[torchao fp8 dqact] skipped (--no-quantize)")

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

    # SigLIP expects fp32 activations; fp16 weights + fp32 input segfaults on Thor.
    vision = vision.float()

    # step 2: vision
    visual = GridVisionExportModule(
        vision_model=vision,                 # paligemma.vision_tower
        projector=projector,                 # paligemma.multi_modal_projector
        sample_pixel_values=pixel_values.float(),
        select_layer=-1,                     # PI05 has no eagle.select_layer
        pixel_shuffle=False,
        downsample_ratio=0.5,
        force_float32_input=True,            # PI05 vision tower runs fp32 internally
        vision_kwargs={},
    ).eval().to(device=device)

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
    print(f"vision eager execute: {vision_eager_elapsed_ms:.3f} ms")
    print_mem("after vision eager")

    vision_export_s = 0.0
    vision_compile_s = 0.0
    vision_export_compile_s = 0.0
    vision_trt_elapsed_ms = 0.0
    vision_stage_buckets = {
        "export_aot": 0.0,
        "run_decompositions": 0.0,
        "post_lowering_partition": 0.0,
        "engine_build": 0.0,
    }
    embs_trt = embs_eager
    embs_eager_plugin = None
    trt_engine = None
    exported = None
    hidden_states = None

    if skip_trt:
        print("Skipping vision TRT (--skip-trt)")
    else:
        # --- Patch SigLIP attention -> ViTPluginAttention ---
        hidden_states = vision.embeddings(pixel_values.float())
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        patched = patch_vision_attention(
            vision,
            batch_size=batch_size,
            seq_len=seq_len,
            name="SigLIP",
        )
        try:
            # --- Rung B: eager with plugin attention (usually invalid eagerly) ---
            with torch.no_grad():
                embs_eager_plugin = visual(pixel_values)

            # --- Rung C: TRT compiled from patched module ---
            print("Compiling vision (stock export + compile)")
            with stage_timing() as timer:
                t_export0 = time.perf_counter()
                exported = torch.export.export(visual, args=(pixel_values,), strict=False)
                vision_export_s = time.perf_counter() - t_export0
                input_specs = make_input_spec((pixel_values,))
                t_compile0 = time.perf_counter()
                trt_engine = torch_tensorrt.dynamo.compile(
                    exported,
                    inputs=input_specs,
                    **VISION_TRT_SETTINGS,
                )
                vision_compile_s = time.perf_counter() - t_compile0
            vision_export_compile_s = vision_export_s + vision_compile_s
            vision_stage_buckets = print_stage_breakdown(
                "vision stock",
                export_seconds=vision_export_s,
                compile_seconds=vision_compile_s,
                snapshot=timer.snapshot(),
            )
            print(
                f"[vision stock] export={vision_export_s:.3f}s  "
                f"compile={vision_compile_s:.3f}s  "
                f"export+compile={vision_export_compile_s:.3f}s"
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

        parity("PI05 vision A vs C", embs_eager, embs_trt)

    # step 3 language
    print("Compiling language" if not skip_trt else "Running language (eager, --skip-trt)")
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
    lm_dtype = quant_compute_dtype
    inputs_embeds = inputs_embeds.to(device=device, dtype=lm_dtype).contiguous()

    # free the vision engine + export artifacts before the language TRT build so
    # TensorRT has enough contiguous GPU memory for its builder allocation.
    free_cuda_memory(
        trt_engine,
        exported,
        visual,
        embs_trt,
        embs_eager,
        embs_eager_plugin,
        trt_image_embs,
        hidden_states,
        pixel_values,
    )
    # Vision TRT is done; keep only the language stack on GPU until diffusion.
    vision.cpu()
    paligemma.multi_modal_projector.cpu()
    model.paligemma_with_expert.gemma_expert.cpu()
    free_cuda_memory()

    # Time eager language here, before the TRT builder allocates GPU: the full
    # language weights are still resident and this is the only window where the
    # eager path fits alongside nothing else on memory-tight Thor.
    def _run_eager_language():
        return language(
            inputs_embeds=inputs_embeds.to(dtype=lm_dtype),
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            output_hidden_states=False,
            return_dict=True,
        )

    with torch.no_grad():
        eager_out = _run_eager_language()
        for _ in range(5):
            _run_eager_language()

        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _run_eager_language()
        end.record()
        torch.cuda.synchronize()
        eager_elapsed_ms = start.elapsed_time(end) / 100

    print(f"lm eager execute: {eager_elapsed_ms:.3f} ms")
    lm_hidden_eager = eager_out.last_hidden_state
    free_cuda_memory(eager_out)
    print_mem("after language eager")

    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(decoder.layers)

    language_compile_s = 0.0
    trt_elapsed_ms = 0.0
    lm_trt_engine = None
    lm = None
    trt_out = None
    flat_tensors = None
    kv_caches = None

    if skip_trt:
        prefix_k = torch.zeros(
            num_layers, bsz, num_key_value_heads, seq_len, head_dim,
            device=device, dtype=quant_compute_dtype,
        )
        prefix_v = torch.zeros_like(prefix_k)
        free_cuda_memory(policy, pre_processor, post_processor, model_inputs, frame, data)
        del policy, pre_processor, post_processor
        free_cuda_memory()
    else:
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
        ds_stack = torch.zeros(0, bsz, seq_len, hidden_size, device=device, dtype=dtype)
        flat_tensors = (
            inputs_embeds,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            last_token_ids,
            ds_stack,
            *kv_caches,
        )

        # PI05 prefix attends bidirectionally; patch_language_attention wires the
        # context attention mask type into the plugin config that the TRT converter
        # reads at compile time.
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
                _, lm_hidden_trt_ref, _, _ = lm(*flat_tensors)

            input_names = list(VLA_LANGUAGE_INPUT_NAMES) + [
                f"past_key_values_{i}" for i in range(num_layers)
            ]
            lm_input_specs = make_language_edge_input_specs(
                input_names,
                flat_tensors,
                batch_size=bsz,
                max_seq_len=seq_len,
                static_prefill_seq_len=True,
            )
            free_cuda_memory(policy, pre_processor, post_processor, model_inputs, frame, data)
            del policy, pre_processor, post_processor
            free_cuda_memory()

            lang_engine_dir = Path(os.environ.get("ENGINE_DIR", "/tmp/pi05_edge_llm")) / "language_e2e"
            t_lang0 = time.perf_counter()
            save_trt_engine_module(
                lm,
                flat_tensors,
                lang_engine_dir,
                engine_file="language.engine",
                model_type="language",
                component="language",
                input_names=input_names,
                output_names=list(VLA_LANGUAGE_OUTPUT_NAMES),
                extra_config={
                    **language_edge_llm_config(
                        cfg,
                        max_seq_len=seq_len,
                        batch_size=bsz,
                        num_layers=num_layers,
                    ),
                    "context_attention_mask_type": int(ContextAttentionMaskType.PADDING),
                },
                input_specs=lm_input_specs,
                flat_tensors=flat_tensors,
                trt_settings=language_edge_trt_settings(offload_module_to_cpu=False),
            )
            language_compile_s = time.perf_counter() - t_lang0
            print(f"[language stock] save_trt_engine_module={language_compile_s:.3f}s")
            free_cuda_memory(lm)
            lm_trt_engine = SerializedPi05Language(SerializedTRTEngine(lang_engine_dir))

            for _ in range(5):
                with torch.no_grad():
                    lm_trt_engine(*flat_tensors)

            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                with torch.no_grad():
                    trt_out = lm_trt_engine(*flat_tensors)
            end.record()
            torch.cuda.synchronize()
            trt_elapsed_ms = start.elapsed_time(end) / 100
        finally:
            restore_attention(patched)

        parity("PI05 language A vs C (TRT)", lm_hidden_eager, trt_out[1])
        prefix_k = trt_out[2].to(device=device, dtype=quant_compute_dtype).contiguous()
        prefix_v = trt_out[3].to(device=device, dtype=quant_compute_dtype).contiguous()
        print("Releasing language TRT runtime before diffusion compile")
        release_serialized_trt_engine(lm_trt_engine)

    free_cuda_memory(
        lm_trt_engine,
        lm,
        trt_out,
        flat_tensors,
        kv_caches,
        inputs_embeds,
        lm_hidden_eager,
        language,
        lm_head,
    )
    model.cpu()
    free_cuda_memory()
    move_pi05_diffusion_modules_to_device(model, device, dtype)
    force_hf_attention(model.paligemma_with_expert.gemma_expert.model, "eager")

    if quantize:
        quantize_fp8_dqact(model.paligemma_with_expert.gemma_expert, "gemma_expert")
        for _name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            quantize_fp8_dqact(getattr(model, _name), _name)
        print_module_param_mem("after diffusion fp8 dqact", _param_parts())
        print_mem("after diffusion fp8 dqact", model)
    else:
        print_module_param_mem("diffusion fp16 (no quant)", _param_parts())
        print_mem("diffusion fp16 (no quant)", model)

    print("Compiling diffusion" if not skip_trt else "Running diffusion (eager, --skip-trt)")
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
        dtype=quant_compute_dtype,
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

    diffusion_eager_elapsed_ms = 0.0
    try:
        with torch.no_grad():
            eager_velocity = diffusion_model(*diffusion_input)
            for _ in range(5):
                diffusion_model(*diffusion_input)

            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                diffusion_model(*diffusion_input)
            end.record()
            torch.cuda.synchronize()
            diffusion_eager_elapsed_ms = start.elapsed_time(end) / 100
        print(f"diffusion eager execute: {diffusion_eager_elapsed_ms:.3f} ms")
        print_mem("after diffusion eager")
    except torch.cuda.OutOfMemoryError as exc:
        print(f"diffusion eager OOM with quantization: {exc}")
        print_mem("after diffusion eager OOM")
        eager_velocity = None
        free_cuda_memory()

    diffusion_export_s = 0.0
    diffusion_compile_s = 0.0
    diffusion_export_compile_s = 0.0
    diffusion_trt_elapsed_ms = 0.0
    diffusion_stage_buckets = {
        "export_aot": 0.0,
        "run_decompositions": 0.0,
        "post_lowering_partition": 0.0,
        "engine_build": 0.0,
    }
    diffusion_trt_engine = None
    diffusion_exported = None

    if skip_trt or eager_velocity is None:
        if eager_velocity is None:
            print("Skipping diffusion TRT (eager path failed)")
        else:
            print("Skipping diffusion TRT (--skip-trt)")
    else:
        print("Compiling diffusion (stock export + compile)")
        with stage_timing() as timer:
            t_export0 = time.perf_counter()
            diffusion_exported = torch.export.export(diffusion_model, args=diffusion_input, strict=False)
            diffusion_export_s = time.perf_counter() - t_export0
            diffusion_input_specs = make_input_spec(diffusion_input)
            t_compile0 = time.perf_counter()
            diffusion_trt_engine = torch_tensorrt.dynamo.compile(
                diffusion_exported,
                inputs=diffusion_input_specs,
                **ACTION_TRT_SETTINGS,
            )
            diffusion_compile_s = time.perf_counter() - t_compile0
        diffusion_export_compile_s = diffusion_export_s + diffusion_compile_s
        diffusion_stage_buckets = print_stage_breakdown(
            "diffusion stock",
            export_seconds=diffusion_export_s,
            compile_seconds=diffusion_compile_s,
            snapshot=timer.snapshot(),
        )
        print(
            f"[diffusion stock] export={diffusion_export_s:.3f}s  "
            f"compile={diffusion_compile_s:.3f}s  "
            f"export+compile={diffusion_export_compile_s:.3f}s"
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

        parity("PI05 diffusion A vs C (TRT)", eager_velocity, trt_velocity)

    eager_total_ms = vision_eager_elapsed_ms + eager_elapsed_ms + diffusion_eager_elapsed_ms
    trt_total_ms = vision_trt_elapsed_ms + trt_elapsed_ms + diffusion_trt_elapsed_ms

    def _speedup(eager_ms: float, trt_ms: float) -> str:
        if eager_ms <= 0.0 or trt_ms <= 0.0:
            return "n/a (benchmark skipped)"
        return f"{eager_ms / trt_ms:.3f}x"

    print()
    print("=== stock compile timings (compare to one-shot script) ===")
    print(
        f"  {'vision':24s}  export={vision_export_s:7.3f}s  "
        f"compile={vision_compile_s:7.3f}s  total={vision_export_compile_s:7.3f}s"
    )
    print(f"  {'language_stock_save':24s}  total={language_compile_s:7.3f}s")
    print(
        f"  {'diffusion':24s}  export={diffusion_export_s:7.3f}s  "
        f"compile={diffusion_compile_s:7.3f}s  total={diffusion_export_compile_s:7.3f}s"
    )
    print(
        f"  {'vision+diffusion stock':24s}  "
        f"total={vision_export_compile_s + diffusion_export_compile_s:7.3f}s"
    )
    print()
    print("=== stock stage focus (vision + diffusion) ===")
    for name, buckets in (
        ("vision", vision_stage_buckets),
        ("diffusion", diffusion_stage_buckets),
    ):
        print(
            f"  {name:10s}  "
            f"export={buckets['export_aot']:6.3f}s  "
            f"decomp={buckets['run_decompositions']:6.3f}s  "
            f"lower+part={buckets['post_lowering_partition']:6.3f}s  "
            f"engine={buckets['engine_build']:6.3f}s"
        )

    _, _, peak_mib = cuda_mem_mib()
    print()
    print("=== eager metrics (quant vs fp16) ===")
    print(f"  quantize={quantize}  skip_trt={skip_trt}")
    print(
        f"  full_model params: {_mib(total_param_nbytes(model)):.1f} MiB "
        f"({format_dtype_mib(param_nbytes_by_dtype(model))})"
    )
    print(f"  cuda peak allocated: {peak_mib:.1f} MiB")
    print(f"  vision eager execute: {vision_eager_elapsed_ms:.3f} ms")
    print(f"  vision trt execute: {vision_trt_elapsed_ms:.3f} ms")
    print(f"  vision speedup: {_speedup(vision_eager_elapsed_ms, vision_trt_elapsed_ms)}")
    print(f"  lm eager execute: {eager_elapsed_ms:.3f} ms")
    print(f"  lm trt execute: {trt_elapsed_ms:.3f} ms")
    print(f"  lm speedup: {_speedup(eager_elapsed_ms, trt_elapsed_ms)}")
    print(f"  diffusion eager execute: {diffusion_eager_elapsed_ms:.3f} ms")
    print(f"  diffusion trt execute: {diffusion_trt_elapsed_ms:.3f} ms")
    print(f"  diffusion speedup: {_speedup(diffusion_eager_elapsed_ms, diffusion_trt_elapsed_ms)}")
    print(f"  total eager execute: {eager_total_ms:.3f} ms")
    print(f"  total trt execute: {trt_total_ms:.3f} ms")
    print(f"  total speedup: {_speedup(eager_total_ms, trt_total_ms)}")

    free_cuda_memory(
        diffusion_trt_engine,
        diffusion_exported,
        diffusion_model,
    )

    return 0

if __name__ == "__main__":
    SystemExit(main())