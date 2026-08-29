"""PI05 e2e identical to ``test_vla_pi05_e2e.py``, but compile via one-shot APIs.

Stock path (original script)::

    ep = torch.export.export(mod, args=...)
    engine = torch_tensorrt.dynamo.compile(ep, inputs=..., **settings)

This script::

    ep = torch_tensorrt.dynamo.export_for_tensorrt(mod, args=..., ...)
    engine = torch_tensorrt.dynamo.compile(
        ep, inputs=..., skip_decompositions=True, **settings
    )

Language still goes through ``save_trt_engine_module`` (dynamic-shape Edge export);
that path is timed but not yet switched to one-shot.

Run both scripts and compare the printed ``*_compile_seconds`` lines.
"""

from __future__ import annotations

import os
import sys
import logging
import torch_tensorrt
import torch
import time

from pathlib import Path

if hasattr(getattr(torch_tensorrt, "logging", None), "set_level"):
    torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import OBS_IMAGES
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from trt.plugin.plugin_utils import restore_attention
from trt.data import load_test_data, frame_from_test_data
from trt.modules.export.vision import GridVisionExportModule
from trt.modules.export.language import CausalLMExportModule
from trt.modules.export.diffusion import (
    PI05PrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)

from trt.measure import parity
from trt.utils import (
    configure_thor_pytorch,
    force_hf_attention,
    free_cuda_memory,
    move_pi05_diffusion_modules_to_device,
    release_serialized_trt_engine,
)

configure_thor_pytorch()
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.rope import make_rope_rotary_cos_sin

from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import patch_vision_attention, patch_language_attention
from trt.compile import make_input_spec, save_trt_engine_module
from trt.compile_stage_timing import print_stage_breakdown, stage_timing
from trt.executor.models.pi05.load.serialize import SerializedPi05Language
from trt.io_spec import VLA_LANGUAGE_INPUT_NAMES, VLA_LANGUAGE_OUTPUT_NAMES
from trt.language import language_edge_llm_config, language_edge_trt_settings, make_language_edge_input_specs
from trt.serialize import SerializedTRTEngine

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
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
}


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def compile_one_shot(
    module: torch.nn.Module,
    args: tuple,
    settings: dict,
    *,
    label: str,
) -> tuple[torch.nn.Module, dict[str, float], dict[str, float]]:
    """One-shot export+decomp, then compile with ``skip_decompositions=True``."""
    from torch_tensorrt.dynamo import export_for_tensorrt

    module = module.eval()
    input_specs = make_input_spec(args)
    decomp_keys = (
        "enable_experimental_decompositions",
        "decompose_attention",
        "use_distributed_mode_trace",
        "use_fp32_acc",
    )
    export_decomp_kwargs = {k: settings[k] for k in decomp_keys if k in settings}

    _cuda_sync(next(module.parameters()).device)
    with stage_timing() as timer:
        t0 = time.perf_counter()
        exported = export_for_tensorrt(
            module,
            args=args,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
            **export_decomp_kwargs,
        )
        export_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            skip_decompositions=True,
            **settings,
        )
        compile_s = time.perf_counter() - t1
    total_s = export_s + compile_s

    stage_buckets = print_stage_breakdown(
        f"{label} one-shot",
        export_seconds=export_s,
        compile_seconds=compile_s,
        snapshot=timer.snapshot(),
    )
    timings = {
        "export_seconds": export_s,
        "compile_seconds": compile_s,
        "export_plus_compile_seconds": total_s,
    }
    print(
        f"[{label} one-shot] export={export_s:.3f}s  "
        f"compile(skip_decomp)={compile_s:.3f}s  "
        f"export+compile={total_s:.3f}s"
    )
    return engine, timings, stage_buckets


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
        max_state_dim=32,  # PI05 default is 32, not 64
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()

    dtype = torch.float16
    compile_timings: dict[str, dict[str, float]] = {}

    # step 1 - load policy, retrieve vision, lm diffusion,
    # create processors, data sample and replace attention
    config, policy = load_config(device)
    model = policy.model.to(device=device, dtype=dtype).eval()
    paligemma = model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    language = paligemma.language_model

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
        vision_model=vision,  # paligemma.vision_tower
        projector=projector,  # paligemma.multi_modal_projector
        sample_pixel_values=pixel_values.float(),
        select_layer=-1,  # PI05 has no eagle.select_layer
        pixel_shuffle=False,
        downsample_ratio=0.5,
        force_float32_input=True,  # PI05 vision tower runs fp32 internally
        vision_kwargs={},
    ).eval().to(device=device)

    # --- Rung A: eager SDPA (UNPATCHED) ---
    with torch.no_grad():
        embs_eager = visual(pixel_values)

    for _ in range(5):
        visual(pixel_values)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        visual(pixel_values)
    end.record()
    torch.cuda.synchronize()
    vision_eager_elapsed_ms = start.elapsed_time(end) / 100

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

        # --- Rung C: TRT compiled from patched module (one-shot) ---
        print("Compiling vision (one-shot export_for_tensorrt)")
        trt_engine, vision_compile_timings, vision_stage_buckets = compile_one_shot(
            visual,
            (pixel_values,),
            VISION_TRT_SETTINGS,
            label="vision",
        )
        compile_timings["vision"] = vision_compile_timings
        with torch.no_grad():
            embs_trt = trt_engine(pixel_values)

        for _ in range(5):
            trt_engine(pixel_values)

        torch.cuda.synchronize(device)
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
    print("Compiling language (stock save_trt_engine_module; not one-shot yet)")
    lm_head = model.paligemma_with_expert.paligemma.lm_head
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
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    lm_dtype = next(language.parameters()).dtype

    # free the vision engine + export artifacts before the language TRT build so
    # TensorRT has enough contiguous GPU memory for its builder allocation.
    free_cuda_memory(
        trt_engine,
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

    lm_hidden_eager = eager_out.last_hidden_state
    free_cuda_memory(eager_out)

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

        lang_engine_dir = Path(os.environ.get("ENGINE_DIR", "/tmp/pi05_edge_llm")) / "language_e2e_one_shot"
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
        compile_timings["language_stock_save"] = {
            "export_plus_compile_seconds": language_compile_s,
        }
        print(
            f"[language stock] save_trt_engine_module={language_compile_s:.3f}s "
            f"(not one-shot; dynamic-shape Edge export)"
        )
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

    # step 4 diffusion (no action-context stage for PI05)
    print("Releasing language TRT runtime before diffusion compile")
    prefix_k = trt_out[2].to(device=device, dtype=dtype).contiguous()
    prefix_v = trt_out[3].to(device=device, dtype=dtype).contiguous()

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

    print("Compiling diffusion (one-shot export_for_tensorrt)")
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
        dtype=dtype,
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

    diffusion_trt_engine, diffusion_compile_timings, diffusion_stage_buckets = compile_one_shot(
        diffusion_model,
        diffusion_input,
        ACTION_TRT_SETTINGS,
        label="diffusion",
    )
    compile_timings["diffusion"] = diffusion_compile_timings
    with torch.no_grad():
        trt_velocity = diffusion_trt_engine(*diffusion_input)

    for _ in range(5):
        diffusion_trt_engine(*diffusion_input)

    torch.cuda.synchronize(device)
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
    print("=== one-shot compile timings (compare to stock script wall times) ===")
    for name, t in compile_timings.items():
        export_s = t.get("export_seconds")
        compile_s = t.get("compile_seconds")
        total_s = t["export_plus_compile_seconds"]
        if export_s is not None and compile_s is not None:
            print(
                f"  {name:24s}  export={export_s:7.3f}s  "
                f"compile={compile_s:7.3f}s  total={total_s:7.3f}s"
            )
        else:
            print(f"  {name:24s}  total={total_s:7.3f}s")
    one_shot_export_compile = sum(
        t["export_plus_compile_seconds"]
        for k, t in compile_timings.items()
        if k in ("vision", "diffusion")
    )
    print(f"  {'vision+diffusion one-shot':24s}  total={one_shot_export_compile:7.3f}s")
    print()
    print("=== one-shot stage focus (vision + diffusion) ===")
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

    print()
    print(f"vision eager execute: {vision_eager_elapsed_ms:.3f} ms")
    print(f"vision trt execute: {vision_trt_elapsed_ms:.3f} ms")
    print(f"vision speedup: {_speedup(vision_eager_elapsed_ms, vision_trt_elapsed_ms)}")
    print(f"lm eager execute: {eager_elapsed_ms:.3f} ms")
    print(f"lm trt execute: {trt_elapsed_ms:.3f} ms")
    print(f"lm speedup: {_speedup(eager_elapsed_ms, trt_elapsed_ms)}")
    print(f"diffusion eager execute: {diffusion_eager_elapsed_ms:.3f} ms")
    print(f"diffusion trt execute: {diffusion_trt_elapsed_ms:.3f} ms")
    print(f"diffusion speedup: {_speedup(diffusion_eager_elapsed_ms, diffusion_trt_elapsed_ms)}")
    print(f"total eager execute: {eager_total_ms:.3f} ms")
    print(f"total trt execute: {trt_total_ms:.3f} ms")
    print(f"total speedup: {_speedup(eager_total_ms, trt_total_ms)}")

    free_cuda_memory(
        diffusion_trt_engine,
        diffusion_model,
    )

    return 0


if __name__ == "__main__":
    SystemExit(main())
