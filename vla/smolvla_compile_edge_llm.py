"""Export SmolVLA TensorRT engines for TensorRT-Edge-LLM-style deployment.

Layout mirrors ``pi05_compile_edge_llm.py`` / ``groot_compile_edge_llm.py``:

  engine_root/visual/visual.engine
  engine_root/language/language.engine
  engine_root/action/diffusion.engine
  engine_root/fixtures/

Vision and language engines use the shared Edge-LLM VitRunner + PluginLM paths
(``VisualFixedInput``, ``PluginLMForCausalLM`` / ``VLA_LANGUAGE_IO``). Action
diffusion reuses PI0.5-style prefix-KV bindings (``PI05_EDGE_IO.action``).

Export produces ``runtime_smoke/input.json`` for C++ ``llm_inference`` smoke tests.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pathlib
import time
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks, pad_tensor
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from trt.action_rollout import ActionRolloutContext, PrefixKVFlowActionAdapter, sample_actions_raw
from trt.compile import dump_edge_fixture, save_trt_engine_module
from trt.data import load_test_data, prepare_policy_batch
from trt.diffusion import SmolVLAPrefixKVStepEncoder, StaticActionVelocityStep
from trt.io_spec import (
    PI05_ACTION_ROLLOUT,
    PI05_EDGE_IO,
    PipelineIOSpec,
    action_rollout_extra_config,
)
from trt.language import (
    language_edge_llm_config,
    language_edge_output_names,
    language_head_dim,
    make_language_edge_flat_tensors,
    make_language_edge_input_specs,
    make_plugin_lm_causal_wrapper,
    run_prefix_language_eager,
    save_embedding_table,
)
from trt.edge_llm_runtime import (
    run_llm_inference_runtime_smoke,
    save_tokenizer_for_edge_llm,
    write_llm_runtime_smoke_case,
)
from trt.measure import (
    compare_language,
    compute_action_parity_metrics,
    mean,
    print_action_metrics,
    print_timing,
    tensor_error_metrics,
)
from trt.packing import PackedLanguageInputs
from trt.plugin_utils import (
    infer_smolvlm_seq_len,
    load_plugins_for_trt,
    patch_vision_attention,
    restore_attention,
)
from trt.rope import make_dummy_rope_rotary_cos_sin, make_rope_rotary_cos_sin
from trt.serialize import (
    SerializedModuleSpec,
    SerializedPI05Action,
    SerializedPI05Language,
    SerializedTRTEngine,
    load_serialized_modules,
)
from trt.utils import clone_hf_module_for_export, ensure_smolvla_on_device, free_cuda_memory, load_policy, make_smolvla_runner_inputs
from trt.vision import (
    VIT_ENGINE_INPUT_NAME,
    VisualFixedInput,
    nchw_to_hwc,
    vit_visual_edge_config,
)

MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
SEED = 42
WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_EDGE_LLM_PLUGIN_SO = (
    WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/libNvInfer_edgellm_plugin.so"
)
DEFAULT_LLM_INFERENCE_BIN = (
    WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference"
)

VISION_TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
}

LANGUAGE_TRT_SETTINGS = {
    **VISION_TRT_SETTINGS,
}

ACTION_TRT_SETTINGS = {
    **VISION_TRT_SETTINGS,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SmolVLA TensorRT engines for TensorRT-Edge-LLM")

    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="SmolVLA policy/model id to load.")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID, help="LeRobot dataset id used for representative inputs.")
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index used for the compile sample.")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index used for the compile sample.")
    parser.add_argument("--engine-dir", type=str, default="/tmp/smolvla_edge_llm", help="Root directory for exported SmolVLA engines.")
    parser.add_argument("--device", type=str, default="cuda", help="Compile device.")
    parser.add_argument(
        "--llm-inference-bin",
        type=str,
        default=str(DEFAULT_LLM_INFERENCE_BIN),
        help="Path to TensorRT-Edge-LLM llm_inference binary for C++ smoke tests.",
    )

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for compile/test tensors.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Static language length override. Must match the SmolVLA prefix length.")
    parser.add_argument("--num-traj-samples", type=int, default=1, help="Compatibility flag; SmolVLA uses one sampled action rollout.")
    parser.add_argument("--max-generation-length", type=int, default=256, help="Compatibility flag; SmolVLA prefix prefill does not generate tokens.")

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


def action_output_dim(policy: Any) -> int:
    output_feature = policy.config.output_features.get(ACTION)
    if output_feature is None:
        return int(policy.model.config.max_action_dim)
    return int(output_feature.shape[0])


def crop_policy_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    return actions[..., : action_output_dim(policy)]


def build_smolvla_vitrunner_chat_template(tokenizer, *, image_token_id: int) -> dict[str, Any]:
    """Minimal processed_chat_template.json for VitRunner image placeholder expansion."""
    image_format = tokenizer.decode([int(image_token_id)])
    if not image_format.strip():
        image_format = "<image>"
    return {
        "model_path": "smolvla-vitrunner",
        "roles": {
            "user": {"prefix": "", "suffix": ""},
        },
        "content_types": {
            "image": {"format": image_format},
        },
        "generation_prompt": "",
        "default_system_prompt": "",
    }


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


def _smolvla_task_text(batch: dict[str, Any]) -> str:
    task_text = batch.get("task", "")
    if isinstance(task_text, (list, tuple)):
        task_text = task_text[0] if task_text else "pick up the object"
    task_text = str(task_text)
    if task_text and not task_text.endswith("\n"):
        task_text += "\n"
    return task_text


def write_smolvla_runtime_smoke_case(
    engine_root: str | pathlib.Path,
    *,
    task_text: str,
    image: torch.Tensor,
    max_generate_length: int = 0,
) -> pathlib.Path:
    """Write llm_inference JSON + PNG under runtime_smoke/."""
    return write_llm_runtime_smoke_case(
        engine_root,
        task_text=task_text,
        images=[_tensor_image_to_pil(image)],
        max_generate_length=max_generate_length,
    )


def validate_language_len(compact_prefix: PackedLanguageInputs, max_seq_len: int | None) -> int:
    prefix_len = int(compact_prefix.inputs_embeds.shape[1])
    if max_seq_len is not None and int(max_seq_len) != prefix_len:
        raise ValueError(
            "SmolVLA Edge export uses a static prefix. "
            f"--max-seq-len must match prefix length {prefix_len}, got {max_seq_len}."
        )
    return prefix_len


# ---------------------------------------------------------------------------
# SmolVLA prefix packing + action (policy-specific; vision/language use shared trt)
# ---------------------------------------------------------------------------


class _SmolVLAActionExpert(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert

    def forward(self, **kwargs):
        return self.vlm_with_expert.forward(**kwargs)


def make_smolvla_action_step(core) -> StaticActionVelocityStep:
    return StaticActionVelocityStep(
        step_encoder=SmolVLAPrefixKVStepEncoder(core),
        action_expert=_SmolVLAActionExpert(core),
        velocity_decoder=core.action_out_proj,
        output_tokens=int(core.config.chunk_size),
    ).eval()


class SmolVLAVisionEngineAdapter:
    """Serialized VitRunner wrapper: NCHW policy tensors -> [B, S, H] image embeds."""

    def __init__(self, engine: SerializedTRTEngine, *, num_tokens: int):
        self.engine = engine
        self.num_tokens = int(num_tokens)

    def __call__(self, pixel_values_nchw: torch.Tensor) -> torch.Tensor:
        hwc = nchw_to_hwc(pixel_values_nchw.contiguous())
        flat = self.engine({VIT_ENGINE_INPUT_NAME: hwc})[0]
        batch_size = int(pixel_values_nchw.shape[0])
        hidden = flat.shape[-1]
        expected_rows = batch_size * self.num_tokens
        if flat.shape[0] != expected_rows:
            raise ValueError(
                f"Visual engine returned {tuple(flat.shape)} rows, expected {expected_rows}"
            )
        return flat.reshape(batch_size, self.num_tokens, hidden)


class SerializedSmolVLAVision:
    def __init__(self, engine: SerializedTRTEngine):
        builder_config = engine.config.get("builder_config", {})
        seq_len = builder_config.get("seq_len", engine.config.get("seq_len"))
        if seq_len is None:
            raise KeyError(f"Missing vision seq_len in engine config: {engine.engine_dir}")
        self._adapter = SmolVLAVisionEngineAdapter(engine, num_tokens=int(seq_len))

    def __call__(self, pixel_values_nchw: torch.Tensor) -> torch.Tensor:
        return self._adapter(pixel_values_nchw)


def _run_serialized_language(
    language_runner: SerializedPI05Language,
    prefix_embs: torch.Tensor,
    *,
    max_seq_len: int,
    num_layers: int,
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
        for _ in range(int(num_layers))
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
def prepare_smolvla_batch(policy, batch, device: torch.device):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
    images = [img.to(device=device, dtype=torch.float32) for img in images]
    img_masks = [mask.to(device=device) for mask in img_masks]
    state = state.to(device=device, dtype=torch.float32)
    return images, img_masks, tokens, masks, state


@torch.no_grad()
def prepare_smolvla_prefix(
    core,
    images,
    img_masks,
    tokens,
    masks,
    state,
    *,
    visual_runner=None,
) -> PackedLanguageInputs:
    embs = []
    pad_masks = []
    att_masks = []

    for img, img_mask in zip(images, img_masks, strict=False):
        if core.add_image_special_tokens:
            image_start_token = (
                core.vlm_with_expert.embed_language_tokens(
                    core.global_image_start_token.to(device=tokens.device)
                )
                .unsqueeze(0)
                .expand(img.shape[0], -1, -1)
            )
            image_start_mask = torch.ones_like(
                image_start_token[:, :, 0],
                dtype=torch.bool,
                device=image_start_token.device,
            )
            embs.append(image_start_token)
            pad_masks.append(image_start_mask)
            att_masks += [0] * image_start_mask.shape[1]

        if visual_runner is None:
            img_emb = core.vlm_with_expert.embed_image(img)
        else:
            img_emb = visual_runner(img)

        img_emb = img_emb * torch.tensor(
            img_emb.shape[-1] ** 0.5,
            dtype=img_emb.dtype,
            device=img_emb.device,
        )

        batch_size, num_img_embs = img_emb.shape[:2]
        img_mask = img_mask[:, None].expand(batch_size, num_img_embs)
        embs.append(img_emb)
        pad_masks.append(img_mask)
        att_masks += [0] * num_img_embs

        if core.add_image_special_tokens:
            image_end_token = (
                core.vlm_with_expert.embed_language_tokens(
                    core.image_end_token.to(device=tokens.device)
                )
                .unsqueeze(0)
                .expand(img.shape[0], -1, -1)
            )
            image_end_mask = torch.ones_like(
                image_end_token[:, :, 0],
                dtype=torch.bool,
                device=image_end_token.device,
            )
            embs.append(image_end_token)
            pad_masks.append(image_end_mask)
            att_masks += [0] * image_end_mask.shape[1]

    lang_emb = core.vlm_with_expert.embed_language_tokens(tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    embs.append(lang_emb)
    pad_masks.append(masks)
    att_masks += [0] * lang_emb.shape[1]

    state_emb = core.state_proj(state)
    state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
    embs.append(state_emb)
    pad_masks.append(torch.ones(state_emb.shape[:2], dtype=torch.bool, device=state_emb.device))
    att_masks += [1] * state_emb.shape[1]

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)[
        None, :
    ].expand(prefix_embs.shape[0], -1)

    if core.prefix_length > 0 and prefix_pad_masks.shape[1] < core.prefix_length:
        prefix_embs = pad_tensor(prefix_embs, core.prefix_length, pad_value=0)
        prefix_pad_masks = pad_tensor(prefix_pad_masks, core.prefix_length, pad_value=0)
        prefix_att_masks = pad_tensor(prefix_att_masks, core.prefix_length, pad_value=0)

    prefix_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    return PackedLanguageInputs(
        inputs_embeds=prefix_embs,
        pad_mask=prefix_pad_masks,
        attention_mask=prefix_attention_mask,
        position_ids=prefix_position_ids,
    )


def make_action_compile_inputs(core, action_module, prefix_pad_masks, prefix_k, prefix_v, device):
    dtype = next(action_module.step_encoder.action_in_proj.parameters()).dtype
    batch_size = prefix_pad_masks.shape[0]
    x_t = torch.randn(
        batch_size,
        core.config.chunk_size,
        core.config.max_action_dim,
        device=device,
        dtype=dtype,
    )
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    return make_smolvla_runner_inputs(
        core,
        prefix_pad_masks,
        prefix_k,
        prefix_v,
        x_t,
        timestep,
        device,
        edge_llm=True,
    )


def _smolvla_action_adapter(core) -> PrefixKVFlowActionAdapter:
    return PrefixKVFlowActionAdapter(
        core,
        int(core.config.num_steps),
        runner_inputs_fn=make_smolvla_runner_inputs,
    )


def _smolvla_text_config(core):
    return core.vlm_with_expert.get_vlm_model().config.text_config


def _smolvla_language_model(core):
    return core.vlm_with_expert.get_vlm_model().text_model


def _clone_smolvla_language_for_export(core, device: torch.device, *, dtype=torch.dtype):
    """Clone truncated SmolVLM text stack (num_vlm_layers) for PluginLM export."""
    text_model = _smolvla_language_model(core)
    num_layers = int(core.vlm_with_expert.num_vlm_layers)
    lm_config = copy.deepcopy(text_model.config)
    lm_config.num_hidden_layers = num_layers
    return clone_hf_module_for_export(
        text_model,
        device,
        dtype=dtype,
        config=lm_config,
    )


def _smolvla_image_token_id(core) -> int:
    token = core.fake_image_token
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def save_visual_engine_for_edge_llm(
    core,
    pixel_values: torch.Tensor,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
):
    vlm = core.vlm_with_expert.get_vlm_model()
    text_cfg = vlm.config.text_config
    image_token_id = _smolvla_image_token_id(core)
    vision_dtype = next(vlm.vision_model.parameters()).dtype

    pixel_values_nchw = pixel_values.to(device=device, dtype=vision_dtype).contiguous()
    images_hwc = nchw_to_hwc(pixel_values_nchw)

    vision_model = clone_hf_module_for_export(
        vlm.vision_model,
        device,
        dtype=next(vlm.vision_model.parameters()).dtype,
    )
    connector = clone_hf_module_for_export(
        vlm.connector,
        device,
        dtype=next(vlm.connector.parameters()).dtype,
    )

    visual = VisualFixedInput(
        vision_model=vision_model,
        projector=connector,
        sample_pixel_values=images_hwc,
        select_layer=-1,
    ).eval().to(device)

    with torch.no_grad():
        eager_output = visual(images_hwc)

    batch_size, seq_len = infer_smolvlm_seq_len(vision_model, pixel_values_nchw)

    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SmolVLM",
        allow_attention_mask=True,
    )
    try:
        engine_path = save_trt_engine_module(
            visual,
            (images_hwc,),
            engine_dir,
            engine_file="visual.engine",
            model_type="vit",
            component="vision",
            input_names=list(io.vision.input_names),
            output_names=list(io.vision.output_names),
            example_output=eager_output,
            extra_config=vit_visual_edge_config(
                vocab_size=int(text_cfg.vocab_size),
                image_token_id=image_token_id,
                seq_len=int(visual.output_seq_len),
            ),
            trt_settings=VISION_TRT_SETTINGS,
        )
    finally:
        restore_attention(patched)
        free_cuda_memory(visual, vision_model, connector)

    return engine_path, int(visual.output_seq_len)


@torch.no_grad()
def _run_smolvla_plugin_language_eager(
    core,
    prefix: PackedLanguageInputs,
    device: torch.device,
    *,
    language_model=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager PluginLM prefill matching the exported SmolVLA language.engine."""
    prefix_embs = prefix.inputs_embeds.to(device=device, dtype=torch.float16).contiguous()
    max_seq_len = int(prefix_embs.shape[1])
    batch_size = int(prefix_embs.shape[0])
    cfg = _smolvla_text_config(core)
    num_layers = int(core.vlm_with_expert.num_vlm_layers)

    lm = _smolvla_language_model(core)
    decoder = getattr(lm, "model", lm)
    lm_head = core.vlm_with_expert.vlm.lm_head
    lm_wrapper = make_plugin_lm_causal_wrapper(
        decoder,
        cfg,
        lm_head,
        log_prefix="smolvla",
    ).to(device=device, dtype=torch.float16).eval()

    kv_caches = [
        torch.zeros(
            batch_size,
            2,
            int(cfg.num_key_value_heads),
            max_seq_len,
            language_head_dim(cfg),
            device=device,
            dtype=prefix_embs.dtype,
        )
        for _ in range(num_layers)
    ]
    ctx_len = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.int32)
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        max_seq_len,
        device,
        language_model=language_model,
        position_ids=prefix.position_ids,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (batch_size, 1),
        max_seq_len - 1,
        device=device,
        dtype=torch.int64,
    )

    _, hidden, prefix_k, prefix_v = lm_wrapper(
        prefix_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )
    return hidden, prefix_k, prefix_v


def save_lm_engine_for_edge_llm(
    core,
    prefix: PackedLanguageInputs,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    tokenizer=None,
    io: PipelineIOSpec = PI05_EDGE_IO,
):
    prefix_embs = prefix.inputs_embeds.to(device=device, dtype=torch.float16).contiguous()
    max_seq_len = int(prefix_embs.shape[1])
    batch_size = int(prefix_embs.shape[0])
    image_token_id = _smolvla_image_token_id(core)

    lm = _clone_smolvla_language_for_export(core, device, dtype=torch.float16)
    decoder = getattr(lm, "model", lm)
    cfg = lm.config
    num_layers = int(core.vlm_with_expert.num_vlm_layers)

    lm_head = clone_hf_module_for_export(
        core.vlm_with_expert.vlm.lm_head,
        device,
        dtype=torch.float16,
    )
    lm_wrapper = make_plugin_lm_causal_wrapper(
        decoder,
        cfg,
        lm_head,
        log_prefix="smolvla",
    ).to(device=device).eval()

    sample_inputs, _trace_seq_len = make_language_edge_flat_tensors(
        prefix_embs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        device=device,
        dtype=prefix_embs.dtype,
        static_prefill_seq_len=True,
    )

    input_names = io.language_input_names(num_layers)
    input_specs = make_language_edge_input_specs(
        input_names,
        sample_inputs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        static_prefill_seq_len=True,
    )

    with torch.no_grad():
        example_output = lm_wrapper(*sample_inputs)

    save_embedding_table(lm, engine_dir)
    if tokenizer is not None:
        save_tokenizer_for_edge_llm(
            "",
            engine_dir,
            tokenizer=tokenizer,
            chat_template=build_smolvla_vitrunner_chat_template(
                tokenizer,
                image_token_id=image_token_id,
            ),
        )

    output_names = language_edge_output_names(io.language.output_names, num_layers)

    engine_path = save_trt_engine_module(
        lm_wrapper,
        sample_inputs,
        engine_dir,
        engine_file="language.engine",
        model_type="smolvla",
        component="language",
        input_names=input_names,
        output_names=output_names,
        example_output=example_output,
        extra_config=language_edge_llm_config(
            cfg,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_layers=num_layers,
            image_token_id=image_token_id,
        ),
        input_specs=input_specs,
        flat_tensors=sample_inputs,
        trt_settings=LANGUAGE_TRT_SETTINGS,
    )
    free_cuda_memory(lm, lm_wrapper, lm_head)
    return engine_path


def save_action_engine_for_edge_llm(
    core,
    prefix: PackedLanguageInputs,
    engine_dir: str | pathlib.Path,
    *,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
):
    action_module = make_smolvla_action_step(core).to(device)
    text_cfg = _smolvla_text_config(core)
    prefix_k = torch.zeros(
        int(core.vlm_with_expert.num_vlm_layers),
        int(prefix.inputs_embeds.shape[0]),
        int(text_cfg.num_key_value_heads),
        int(prefix.pad_mask.shape[1]),
        int(text_cfg.head_dim),
        device=device,
        dtype=torch.float16,
    )
    prefix_v = torch.zeros_like(prefix_k)
    sample_inputs = make_action_compile_inputs(
        core,
        action_module,
        prefix.pad_mask,
        prefix_k,
        prefix_v,
        device,
    )
    sample_inputs = tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in sample_inputs)

    with torch.no_grad():
        eager_output = action_module(*sample_inputs)

    return save_trt_engine_module(
        action_module,
        sample_inputs,
        engine_dir,
        engine_file="diffusion.engine",
        model_type="smolvla",
        component="diffusion",
        input_names=list(io.action.input_names),
        output_names=list(io.action.output_names),
        example_output=eager_output,
        extra_config={
            "engine_role": "single_action_denoising_step",
            **action_rollout_extra_config(
                io,
                PI05_ACTION_ROLLOUT,
                num_steps=int(core.config.num_steps),
                chunk_size=int(core.config.chunk_size),
                max_action_dim=int(core.config.max_action_dim),
                prefix_seq_len=int(prefix.pad_mask.shape[1]),
                num_layers=int(core.vlm_with_expert.num_vlm_layers),
                num_key_value_heads=int(text_cfg.num_key_value_heads),
                head_dim=int(text_cfg.head_dim),
            ),
        },
        trt_settings=ACTION_TRT_SETTINGS,
    )


@torch.no_grad()
def compare_smolvla_edge_pipeline_to_eager(
    *,
    core,
    policy,
    images,
    img_masks,
    tokens,
    masks,
    state,
    trt_image_embs,
    trt_hidden: torch.Tensor,
    trt_prefix_k: torch.Tensor,
    trt_prefix_v: torch.Tensor,
    action_runner: SerializedPI05Action,
    device: torch.device,
    seed: int,
):
    set_reproducible_seed(seed, device)
    print("\n=== SmolVLA Edge engine parity vs eager ===")

    ensure_smolvla_on_device(core, device)

    eager_prefix = prepare_smolvla_prefix(
        core, images, img_masks, tokens, masks, state, visual_runner=None
    )
    eager_hidden, eager_prefix_k, eager_prefix_v = _run_smolvla_plugin_language_eager(
        core,
        eager_prefix,
        device,
        language_model=_smolvla_language_model(core),
    )

    for i, (eager_img, trt_img) in enumerate(
        zip(
            [core.vlm_with_expert.embed_image(img) for img in images],
            trt_image_embs,
            strict=True,
        )
    ):
        tensor_error_metrics(f"vision[{i}]", trt_img, eager_img)

    compare_language(
        eager_hidden,
        eager_prefix_k,
        eager_prefix_v,
        trt_hidden,
        trt_prefix_k,
        trt_prefix_v,
        eager_prefix.pad_mask,
    )

    noise = core.sample_noise(
        (state.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )
    eager_actions = core.sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        state,
        noise=noise.clone(),
    )

    trt_actions = sample_actions_raw(
        action_runner,
        ActionRolloutContext(
            noise=noise.clone(),
            device=device,
            prefix_k=trt_prefix_k,
            prefix_v=trt_prefix_v,
            prefix_pad_mask=eager_prefix.pad_mask,
        ),
        _smolvla_action_adapter(core),
    )

    action_dim = action_output_dim(policy)
    eager_actions = eager_actions[:, :, :action_dim]
    trt_actions = trt_actions[:, :, :action_dim]

    metrics = compute_action_parity_metrics(trt_actions, eager_actions)
    print(
        f"  action parity: ADE={metrics['action_ade']:.6f}  "
        f"FDE={metrics['action_fde']:.6f}  "
        f"mean_abs={metrics['mean_abs']:.6f}  "
        f"max_abs={metrics['max_abs']:.6f}"
    )
    return metrics


@torch.no_grad()
def _dump_smolvla_edge_fixture(
    *,
    engine_root: str | pathlib.Path,
    core,
    prefix: PackedLanguageInputs,
    lm_hidden_states: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    pixel_values: torch.Tensor,
    seed: int,
    device: torch.device,
    io: PipelineIOSpec = PI05_EDGE_IO,
) -> pathlib.Path:
    set_reproducible_seed(seed, device)
    noise = core.sample_noise(
        (prefix.inputs_embeds.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )
    timestep = torch.ones(prefix.inputs_embeds.shape[0], device=device, dtype=torch.float32)
    action_module = make_smolvla_action_step(core).to(device)
    prefix_k = prefix_k.to(device=device, dtype=noise.dtype).contiguous()
    prefix_v = prefix_v.to(device=device, dtype=noise.dtype).contiguous()
    noise, timestep, prefix_k, prefix_v, position_ids, attention_mask = make_smolvla_runner_inputs(
        core,
        prefix.pad_mask,
        prefix_k,
        prefix_v,
        noise,
        timestep,
        device,
    )
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
            prefix_pad_mask=prefix.pad_mask,
        ),
        _smolvla_action_adapter(core),
    )
    velocity_name = io.action.output_names[0]
    return dump_edge_fixture(
        engine_root,
        {
            "pixel_values": pixel_values.to(device=device).contiguous(),
            "inputs_embeds": prefix.inputs_embeds.to(device=device, dtype=torch.float16),
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "lm_hidden_states": lm_hidden_states.to(device=device, dtype=torch.float16),
            "initial_actions": noise,
            "timestep": timestep,
            velocity_name: velocity,
            "actions_out": actions_out.to(device=device, dtype=noise.dtype),
        },
    )


def save_edge_engines_for_edge_llm(
    core,
    policy,
    device: torch.device,
    batch: dict[str, Any],
    *,
    seed: int = SEED,
    max_seq_len: int | None = None,
    engine_root: str | pathlib.Path = "/tmp/smolvla_edge_llm",
    io: PipelineIOSpec = PI05_EDGE_IO,
    accuracy_check: bool = True,
    stage_parity: bool = True,
    max_generate_length: int = 0,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict]:
    engine_root = pathlib.Path(engine_root)
    text_cfg = _smolvla_text_config(core)
    language_model = _smolvla_language_model(core)

    images, img_masks, tokens, masks, state = prepare_smolvla_batch(policy, batch, device)
    pixel_values = images[0].contiguous()
    tokenizer = AutoTokenizer.from_pretrained(core.config.vlm_model_name)

    ensure_smolvla_on_device(core, device)
    load_plugins_for_trt()

    print("exporting SmolVLA vision.engine")
    vision_engine_dir = engine_root / "visual"
    vision_engine, vision_seq_len = save_visual_engine_for_edge_llm(
        core,
        pixel_values,
        vision_engine_dir,
        device=device,
        io=io,
    )

    vision_adapter = SmolVLAVisionEngineAdapter(
        SerializedTRTEngine(vision_engine_dir),
        num_tokens=vision_seq_len,
    )
    with torch.no_grad():
        trt_image_embs = [vision_adapter(image.contiguous()) for image in images]
        prefix = prepare_smolvla_prefix(
            core,
            images,
            img_masks,
            tokens,
            masks,
            state,
            visual_runner=vision_adapter,
        )
    free_cuda_memory(vision_adapter)

    language_max_seq_len = validate_language_len(prefix, max_seq_len)

    print("exporting SmolVLA language.engine")
    language_engine_dir = engine_root / "language"
    language_engine = save_lm_engine_for_edge_llm(
        core,
        prefix,
        language_engine_dir,
        device=device,
        tokenizer=tokenizer,
        io=io,
    )
    language_runner = SerializedPI05Language(SerializedTRTEngine(language_engine_dir))
    with torch.no_grad():
        trt_hidden, trt_prefix_k, trt_prefix_v = _run_serialized_language(
            language_runner,
            prefix.inputs_embeds,
            max_seq_len=language_max_seq_len,
            num_layers=int(core.vlm_with_expert.num_vlm_layers),
            device=device,
            position_ids=prefix.position_ids,
            cfg=text_cfg,
            language_model=language_model,
        )

    print("exporting SmolVLA diffusion.engine")
    action_engine_dir = engine_root / "action"
    action_engine = save_action_engine_for_edge_llm(
        core,
        prefix,
        action_engine_dir,
        device=device,
        io=io,
    )
    action_runner = SerializedPI05Action(SerializedTRTEngine(action_engine_dir))

    ensure_smolvla_on_device(core, device)

    if accuracy_check and stage_parity:
        compare_smolvla_edge_pipeline_to_eager(
            core=core,
            policy=policy,
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            state=state,
            trt_image_embs=trt_image_embs,
            trt_hidden=trt_hidden,
            trt_prefix_k=trt_prefix_k,
            trt_prefix_v=trt_prefix_v,
            action_runner=action_runner,
            device=device,
            seed=seed,
        )

    fixture_dir = _dump_smolvla_edge_fixture(
        engine_root=engine_root,
        core=core,
        prefix=prefix,
        lm_hidden_states=trt_hidden,
        prefix_k=trt_prefix_k,
        prefix_v=trt_prefix_v,
        pixel_values=pixel_values,
        seed=seed,
        device=device,
        io=io,
    )

    task_text = _smolvla_task_text(batch)
    smoke_input = write_smolvla_runtime_smoke_case(
        engine_root,
        task_text=task_text,
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
        "vision_output_seq_len": vision_seq_len,
        "language_max_seq_len": language_max_seq_len,
        "prefix_seq_len": int(prefix.pad_mask.shape[1]),
        "chunk_size": int(core.config.chunk_size),
        "max_action_dim": int(core.config.max_action_dim),
        "output_action_dim": action_output_dim(policy),
        "num_inference_steps": int(core.config.num_steps),
        "fixture_dir": str(fixture_dir),
        "runtime_smoke_input": str(smoke_input),
        **io.to_plugin_info(),
    }

    return vision_engine, language_engine, action_engine, plugin_info


@torch.no_grad()
def run_inference_pytorch_smolvla(
    core,
    policy,
    batch: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    set_reproducible_seed(seed, device)
    images, img_masks, tokens, masks, state = prepare_smolvla_batch(policy, batch, device)

    start_time = time.perf_counter()
    noise = core.sample_noise(
        (state.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )
    actions = core.sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        state,
        noise=noise,
    )
    elapsed = time.perf_counter() - start_time
    actions = crop_policy_actions(policy, actions)
    extra = {"noise": noise}
    return actions, extra, elapsed


@torch.no_grad()
def run_inference_smolvla_engines(
    core,
    policy,
    batch: dict[str, Any],
    *,
    vision_runner,
    language_runner: SerializedPI05Language,
    diffusion_runner: SerializedPI05Action,
    plugin_info: dict,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    set_reproducible_seed(seed, device)
    images, img_masks, tokens, masks, state = prepare_smolvla_batch(policy, batch, device)

    start_time = time.perf_counter()
    prefix = prepare_smolvla_prefix(
        core,
        images,
        img_masks,
        tokens,
        masks,
        state,
        visual_runner=vision_runner,
    )
    _, prefix_k, prefix_v = _run_serialized_language(
        language_runner,
        prefix.inputs_embeds,
        max_seq_len=int(plugin_info["language_max_seq_len"]),
        num_layers=int(core.vlm_with_expert.num_vlm_layers),
        device=device,
        position_ids=prefix.position_ids,
        cfg=_smolvla_text_config(core),
        language_model=_smolvla_language_model(core),
    )
    noise = core.sample_noise(
        (state.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )
    actions = sample_actions_raw(
        diffusion_runner,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=prefix.pad_mask,
        ),
        _smolvla_action_adapter(core),
    )
    elapsed = time.perf_counter() - start_time
    actions = crop_policy_actions(policy, actions)
    extra = {
        "noise": noise,
        "prefix_k": prefix_k,
        "prefix_v": prefix_v,
        "prefix_pad_mask": prefix.pad_mask,
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

    policy = load_policy(SmolVLAPolicy, args.model_id, device).to(device).eval()
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
        print(
            "SmolVLA in-memory TRT plugin compile is not wired in this script; "
            "use Test/compile/smolvla_compile_trt.py or pass --skip-trt."
        )

    if not args.skip_engine:
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
                SerializedModuleSpec("vision", "visual", SerializedSmolVLAVision),
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
            pred_actions_pt, _, _ = run_inference_pytorch_smolvla(
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

        if not args.skip_trt and trt_vision is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            pred_actions_trt, _, _ = run_inference_smolvla_engines(
                core,
                policy,
                compile_inputs,
                vision_runner=trt_vision,
                language_runner=trt_lm,
                diffusion_runner=trt_diffusion,
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
                print(
                    f"  TRT Plugin : {trt_elapsed:7.1f} ms   "
                    f"actionADE={trt_metrics['action_ade']:.6f}  "
                    f"mean_abs={trt_metrics['mean_abs']:.6f}"
                )
            else:
                print(f"  TRT Plugin : {trt_elapsed:7.1f} ms")

        if not args.skip_engine:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            pred_actions_engine, _, _ = run_inference_smolvla_engines(
                core,
                policy,
                compile_inputs,
                vision_runner=engine_vision,
                language_runner=engine_lm,
                diffusion_runner=engine_diffusion,
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
                print(
                    f"  Serialized : {engine_elapsed:7.1f} ms   "
                    f"actionADE={engine_metrics['action_ade']:.6f}  "
                    f"mean_abs={engine_metrics['mean_abs']:.6f}"
                )
            else:
                print(f"  Serialized : {engine_elapsed:7.1f} ms")

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations})")
    print("=" * 78)

    if pt_times:
        print_timing("PyTorch SmolVLA", pt_times[args.warmup:])

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