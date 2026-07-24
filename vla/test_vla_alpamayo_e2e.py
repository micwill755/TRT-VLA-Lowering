from __future__ import annotations

import copy
import gc
import logging
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch_tensorrt

from pathlib import Path

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

_ALPAMAYO_SRC = _TEST_ROOT.parent / "alpamayo" / "src"
if _ALPAMAYO_SRC.is_dir() and str(_ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(_ALPAMAYO_SRC))

from alpamayo_r1 import helper

from trt.compile import make_input_spec
from trt.measure import parity
from trt.modules.export.alpamayo_language import build_alpamayo_prefix_embs
from trt.modules.export.language import CausalLMExportModule
from trt.modules.export.alpamayo_lm_plugin import (
    build_alpamayo_rope_cache,
    pack_deepstack_to_ds_stack,
)
from trt.modules.export.alpamayo_vision import VisualFixedGrid, patch_qwen3vl_vision_attention
from trt.modules.export.diffusion import (
    AlpamayoPrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)
from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import (
    load_plugins_for_trt,
    patch_language_attention,
    restore_attention,
)
from trt.utils import force_hf_attention, free_cuda_memory, release_serialized_trt_engine

TRT_SETTINGS = {
    "disable_tf32": False,
    "use_fp32_acc": False,
    "use_explicit_typing": False,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "enabled_precisions": {torch.float16},
}

# Alpamayo needs a larger numerical-precision budget than PI05/GR00T. Its 8B
# Qwen3-VL LM is 36 layers deep and 4096 hidden units wide, so FP16 accumulation
# loses precision over long reductions and the error compounds across layers.
# The attention plugin is not the issue: the drift comes from the surrounding
# QKV/MLP/norm graph compiled by TRT. Smaller/shallower PI05 and GR00T LMs remain
# within the parity tolerance with FP16 accumulation, whereas Alpamayo measured
# rel_l2~0.10 (53% close). Strong typing and FP32 accumulation reduce that to
# rel_l2~0.01 (94% close), at the expected cost of lower LM speedup. These match
# the known-good accuracy settings in alpamayo_r1.trt.lm_plugin.
LANGUAGE_TRT_SETTINGS = {
    "enabled_precisions": {torch.float32},
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "disable_tf32": True,
    "min_block_size": 1,
    "decompose_attention": True,
    # 8B LM (~16GB fp16) + TRT builder will not fit on a 32GB GPU unless torch
    # weights are offloaded during compile. Flip to False on hosts with more VRAM.
    "offload_module_to_cpu": True,
}


ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
}


def load_config(device, model_path: str = "nvidia/Alpamayo-R1-10B", dtype=torch.float16):
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ModuleNotFoundError as exc:
        if exc.name != "hydra":
            raise
        raise RuntimeError(
            "Alpamayo model loading requires hydra-core and the rest of the alpamayo "
            "package dependencies. Run this script with the Alpamayo Python 3.12 "
            "environment, or install dependencies from /home/micwilliams/workspace/alpamayo."
        ) from exc

    model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to(device).eval()
    processor = helper.get_processor(model.tokenizer)
    return model, processor


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()

    dtype = torch.float16

    model, processor = load_config(device, dtype=dtype)
    vlm_model = model.vlm.model
    vision = vlm_model.visual
    language = vlm_model.language_model

    vision = vision.to(device=device, dtype=dtype).eval()
    language = language.to(device=device, dtype=dtype).eval()
    model.expert = model.expert.to(device=device, dtype=dtype).eval()

    force_hf_attention(vision, "eager", use_cache=False)
    force_hf_attention(language, "eager")
    force_hf_attention(model.expert, "eager")

    try:
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    except ModuleNotFoundError as exc:
        if exc.name != "physical_ai_av":
            raise
        raise RuntimeError(
            "Alpamayo e2e data loading requires the physical_ai_av package. "
            "Run this script with the Alpamayo Python 3.12 environment, or install "
            "the alpamayo package dependencies from /home/micwilliams/workspace/alpamayo."
        ) from exc

    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    tokenized_data = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, str(device))

    pixel_values = model_inputs["tokenized_data"]["pixel_values"].to(
        device=device, dtype=dtype
    )
    image_grid_thw = model_inputs["tokenized_data"]["image_grid_thw"].to(device=device)

    # ---------------------------------------------------------------------------
    # STEP 1 — vision
    # ---------------------------------------------------------------------------
    print("Compiling vision")

    visual = VisualFixedGrid(vision, image_grid_thw).to(device=device).eval()

    with torch.no_grad():
        embs_eager, _deepstack_eager = visual(pixel_values, None)

        for _ in range(5):
            visual(pixel_values, None)

        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            visual(pixel_values, None)
        end.record()
        torch.cuda.synchronize()
        vision_eager_elapsed_ms = start.elapsed_time(end) / 100

    if not patch_qwen3vl_vision_attention():
        raise RuntimeError("Failed to patch Qwen3-VL vision attention for TRT export")

    vision_inputs = (pixel_values,)
    with torch.no_grad():
        exported = torch.export.export(visual, args=vision_inputs, strict=False)
        input_specs = make_input_spec(vision_inputs)
        torch.cuda.synchronize(device)
        compile_start = time.perf_counter()
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
        )
        torch.cuda.synchronize(device)
        vision_compile_s = time.perf_counter() - compile_start
        embs_trt, deepstack_trt = trt_engine(*vision_inputs)

    for _ in range(5):
        trt_engine(*vision_inputs)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        trt_engine(*vision_inputs)
    end.record()
    torch.cuda.synchronize()
    vision_trt_elapsed_ms = start.elapsed_time(end) / 100

    parity("vision A vs C (TRT)", embs_eager, embs_trt)

    # ---------------------------------------------------------------------------
    # STEP 2 — language (AttentionPlugin), sequential like PI05 e2e
    # ---------------------------------------------------------------------------
    print("Compiling language")

    # Fuse text + TRT vision features into inputs_embeds / deepstack / RoPE.
    tokenized = copy.deepcopy(model_inputs["tokenized_data"])
    input_ids = tokenized.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )

    image_token_id = int(vlm_model.config.image_token_id)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Like PI05 build_pi05_prefix_embs: embedding table + raw TRT vision tensors.
        inputs_embeds, visual_pos_masks = build_alpamayo_prefix_embs(
            language.embed_tokens,
            input_ids,
            embs_trt,
            image_token_id=image_token_id,
        )
        deepstack_embeds = deepstack_trt
        attn = tokenized.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        try:
            position_ids, rope_deltas = vlm_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw=None, attention_mask=attn
            )
        except (TypeError, IndexError):
            mm_token_type_ids = (input_ids == image_token_id).int()
            position_ids, rope_deltas = vlm_model.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attn,
            )

    bsz = int(inputs_embeds.shape[0])
    seq_len = int(inputs_embeds.shape[1])
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_attention_heads = int(cfg.num_attention_heads)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))

    # Free vision TRT before language (same motivation as PI05).
    free_cuda_memory(
        trt_engine,
        exported,
        input_specs,
        visual,
        embs_trt,
        embs_eager,
        pixel_values,
    )
    vision.cpu()
    vlm_model.visual.cpu()
    free_cuda_memory()

    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    lm_head = model.vlm.lm_head.to(device=device, dtype=dtype).eval()
    decoder = getattr(language, "model", language)
    num_layers = len(decoder.layers)

    # HF eager reference BEFORE patch (PI05 pattern). PluginAttention's torch op
    # body is a zeros stub — not a valid eager reference.
    ds_for_hf: list[torch.Tensor] = []
    for de in (
        list(deepstack_embeds)
        if isinstance(deepstack_embeds, (list, tuple))
        else [deepstack_embeds[i] for i in range(deepstack_embeds.shape[0])]
    ):
        de = de.to(device=device, dtype=dtype)
        if de.dim() > 2:
            de = de.reshape(-1, de.shape[-1])
        ds_for_hf.append(de)

    def _run_eager_language():
        return language(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=ds_for_hf,
            use_cache=False,
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
    eager_last_logits = lm_head(lm_hidden_eager[:, -1]).float()
    free_cuda_memory(eager_out, ds_for_hf)

    ds_stack = pack_deepstack_to_ds_stack(
        deepstack_embeds,
        visual_pos_masks,
        batch_size=bsz,
        max_seq_len=seq_len,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
    free_cuda_memory(deepstack_embeds)

    lm = CausalLMExportModule(
        decoder,
        lm_head,
        select_layer=-1,
    ).eval().to(device=device)

    # Qwen3-VL mRoPE: bake get_rope_index positions into plugin cache (not 1D make_rope_*).
    rope_rotary_cos_sin = build_alpamayo_rope_cache(
        language,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        seq_len=seq_len,
        max_seq_len=seq_len,
        head_dim=head_dim,
        device=device,
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
    flat_tensors = (
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        ds_stack,
        *kv_caches,
    )

    # Causal prefix for Alpamayo (PI05 uses PADDING for bidirectional prefix).
    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        context_attention_mask_type=ContextAttentionMaskType.CAUSAL,
        name="alpamayo-language",
    )
    try:
        with torch.no_grad():
            _, lm_hidden_trt_ref, _, _ = lm(*flat_tensors)

        model.expert.cpu()
        free_cuda_memory(model_inputs, data, messages, lm_hidden_trt_ref)
        free_cuda_memory()

        lm_exported = torch.export.export(lm, args=flat_tensors, strict=False)
        free_cuda_memory()
        torch.cuda.synchronize(device)
        compile_start = time.perf_counter()
        lm_trt_engine = torch_tensorrt.dynamo.compile(
            lm_exported,
            inputs=list(flat_tensors),
            **LANGUAGE_TRT_SETTINGS,
        )
        torch.cuda.synchronize(device)
        language_compile_s = time.perf_counter() - compile_start
        free_cuda_memory(lm_exported)

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

    # PI05 compares hidden states (not full-vocab logits).
    parity("language A vs C (TRT)", lm_hidden_eager, trt_out[1])
    parity("language logits A vs C", eager_last_logits, trt_out[0])

    # Language TRT engine still holds ~8B weights on GPU; drop it before diffusion.
    print("Releasing language TRT runtime before diffusion compile")
    prefix_k = trt_out[2].detach().cpu()
    prefix_v = trt_out[3].detach().cpu()
    release_serialized_trt_engine(lm_trt_engine)
    free_cuda_memory(
        lm_trt_engine,
        lm,
        trt_out,
        flat_tensors,
        kv_caches,
        inputs_embeds,
        ds_stack,
        rope_rotary_cos_sin,
        lm_hidden_eager,
        eager_last_logits,
        language,
        lm_head,
    )
    model.vlm.cpu()
    free_cuda_memory()

    # ---------------------------------------------------------------------------
    # STEP 3 — diffusion (no separate action-context stage for Alpamayo)
    # ---------------------------------------------------------------------------
    print("compiling diffusion")

    n_diffusion_tokens = int(model.action_space.get_action_space_dims()[0])
    action_space_dims = tuple(int(x) for x in model.action_space.get_action_space_dims())

    model.expert.to(device=device, dtype=dtype)
    model.action_in_proj.to(device=device, dtype=dtype)
    model.action_out_proj.to(device=device, dtype=dtype)
    force_hf_attention(model.expert, "eager")
    diffusion_model = StaticActionVelocityStepExportModule(
        step_encoder=AlpamayoPrefixKVStepEncoderExportModule(model),
        action_expert=model.expert,
        velocity_decoder=model.action_out_proj,
        output_tokens=n_diffusion_tokens,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

    prefix_k = prefix_k.to(device=device, dtype=dtype).contiguous()
    prefix_v = prefix_v.to(device=device, dtype=dtype).contiguous()
    prefix_len = int(prefix_k.shape[3])

    step_actions = torch.randn(
        bsz,
        *action_space_dims,
        device=device,
        dtype=dtype,
    )
    step_timestep = torch.zeros(bsz, 1, 1, device=device, dtype=dtype)
    step_position_ids = (
        torch.arange(n_diffusion_tokens, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(3, bsz, -1)
        .clone()
    )
    step_attention_mask = torch.zeros(
        bsz,
        1,
        n_diffusion_tokens,
        prefix_len + n_diffusion_tokens,
        device=device,
        dtype=dtype,
    )

    diffusion_input = (
        step_actions,
        step_timestep,
        prefix_k,
        prefix_v,
        step_position_ids,
        step_attention_mask,
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

    diffusion_exported = torch.export.export(
        diffusion_model, args=diffusion_input, strict=False
    )
    diffusion_input_specs = make_input_spec(diffusion_input)
    torch.cuda.synchronize(device)
    compile_start = time.perf_counter()
    diffusion_trt_engine = torch_tensorrt.dynamo.compile(
        diffusion_exported,
        inputs=diffusion_input_specs,
        **{**ACTION_TRT_SETTINGS, "use_python_runtime": True},
    )
    torch.cuda.synchronize(device)
    diffusion_compile_s = time.perf_counter() - compile_start

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

    parity("diffusion step A vs C (TRT)", eager_velocity, trt_velocity)

    eager_total_ms = vision_eager_elapsed_ms + eager_elapsed_ms + diffusion_eager_elapsed_ms
    trt_total_ms = vision_trt_elapsed_ms + trt_elapsed_ms + diffusion_trt_elapsed_ms
    compile_total_s = vision_compile_s + language_compile_s + diffusion_compile_s

    def _speedup(eager_ms: float, trt_ms: float) -> str:
        return f"{(eager_ms / trt_ms):.3f}x" if trt_ms > 0 else "n/a"

    print(f"vision TRT compile: {vision_compile_s:.3f} s")
    print(f"lm TRT compile: {language_compile_s:.3f} s")
    print(f"diffusion TRT compile: {diffusion_compile_s:.3f} s")
    print(f"total TRT compile: {compile_total_s:.3f} s")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
