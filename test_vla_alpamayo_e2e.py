from __future__ import annotations

import logging
import sys

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
from trt.modules.export.alpamayo_language import (
    Qwen3VLTextModelPrefillExportModule,
    run_vlm_preprocessing,
)
from trt.modules.export.alpamayo_vision import VisualFixedGrid, patch_qwen3vl_vision_attention
from trt.modules.export.diffusion import (
    AlpamayoPrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import force_hf_attention

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
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
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
    model.config.attn_implementation = "sdpa"
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Alpamayo TRT e2e test requires CUDA")

    load_plugins_for_trt()

    dtype = torch.float16

    model, processor = load_config(device, dtype=dtype)
    vlm_model = model.vlm.model
    vision = vlm_model.visual
    language = vlm_model.language_model

    vision.config.attn_implementation = "sdpa"
    vision.config._attn_implementation = "sdpa"
    vision.config.use_cache = False
    vision = vision.to(device=device, dtype=dtype).eval()

    language.config._attn_implementation = "sdpa"
    language = language.to(device=device, dtype=dtype).eval()
    force_hf_attention(language, "eager")

    model.expert.config._attn_implementation = "sdpa"
    model.expert = model.expert.to(device=device, dtype=dtype).eval()
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

    if not patch_qwen3vl_vision_attention():
        raise RuntimeError("Failed to patch Qwen3-VL vision attention for TRT export")

    vision_inputs = (pixel_values, None)
    with torch.no_grad():
        exported = torch.export.export(visual, args=vision_inputs, strict=False)
        input_specs = make_input_spec(vision_inputs)
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
        )
        embs_trt, _ = trt_engine(*vision_inputs)

    parity("vision A vs C (TRT)", embs_eager, embs_trt)

    # ---------------------------------------------------------------------------
    # STEP 2 — language (VLM prefill: fused vision + text -> prefix KV)
    # ---------------------------------------------------------------------------
    print("Compiling language")

    original_visual_forward = vlm_model.visual.forward
    vlm_model.visual.forward = lambda hidden_states, grid_thw=None: trt_engine(
        hidden_states, None
    )
    try:
        with torch.no_grad():
            (
                _input_ids,
                inputs_embeds,
                deepstack_embeds,
                visual_pos_masks,
                position_ids,
                _rope_deltas,
            ) = run_vlm_preprocessing(
                model,
                model_inputs,
                trt_vision=vlm_model.visual,
                device=device,
                dtype=dtype,
            )
    finally:
        vlm_model.visual.forward = original_visual_forward

    bsz = inputs_embeds.shape[0]
    attention_mask = model_inputs["tokenized_data"]["attention_mask"].to(device=device)

    expert_cfg = language.config
    prefill = Qwen3VLTextModelPrefillExportModule(
        language,
        num_layers=int(expert_cfg.num_hidden_layers),
        num_kv_heads=int(expert_cfg.num_key_value_heads),
        head_dim=int(expert_cfg.head_dim),
    ).eval().to(device=device)

    prefill_inputs = (
        attention_mask,
        position_ids.to(device=device, dtype=torch.long),
        inputs_embeds.to(device=device, dtype=dtype).contiguous(),
        visual_pos_masks.to(device=device),
        deepstack_embeds,
    )

    with torch.no_grad():
        lm_hidden_eager, prefix_k, prefix_v = prefill(*prefill_inputs)

    prefill_exported = torch.export.export(prefill, args=prefill_inputs, strict=False)
    prefill_input_specs = make_input_spec(prefill_inputs)
    prefill_trt_engine = torch_tensorrt.dynamo.compile(
        prefill_exported,
        inputs=prefill_input_specs,
        **{**LANGUAGE_TRT_SETTINGS, "use_python_runtime": True},
    )

    with torch.no_grad():
        lm_hidden_trt, _prefix_k_trt, _prefix_v_trt = prefill_trt_engine(*prefill_inputs)

    parity("language A vs C (TRT)", lm_hidden_eager, lm_hidden_trt)

    # ---------------------------------------------------------------------------
    # STEP 3 — diffusion (no separate action-context stage for Alpamayo)
    # ---------------------------------------------------------------------------
    print("compiling diffusion")

    n_diffusion_tokens = int(model.action_space.get_action_space_dims()[0])
    action_space_dims = tuple(int(x) for x in model.action_space.get_action_space_dims())

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

    diffusion_exported = torch.export.export(diffusion_model, args=diffusion_input, strict=False)
    diffusion_input_specs = make_input_spec(diffusion_input)
    diffusion_trt_engine = torch_tensorrt.dynamo.compile(
        diffusion_exported,
        inputs=diffusion_input_specs,
        **{**ACTION_TRT_SETTINGS, "use_python_runtime": True},
    )

    with torch.no_grad():
        trt_velocity = diffusion_trt_engine(*diffusion_input)

    parity("diffusion step A vs C (TRT)", eager_velocity, trt_velocity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
