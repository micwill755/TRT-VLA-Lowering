"""Shared Pi0.5 TensorRT load / prefill / Euler-step helpers for HeiSD and Spec-VLA."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from trt.data import frame_from_test_data, load_test_data
from trt.executor.models.pi05.helpers import (
    build_pi05_prefix_embs,
    make_pi05_suffix_position_and_mask,
    pad_pi05_compact_prefix,
)
from trt.executor.models.pi05.load.serialize import (
    SerializedPi05Action,
    SerializedPi05Language,
    SerializedPi05Vision,
)
from trt.plugin.plugin_utils import create_kv_caches, load_plugins_for_trt
from trt.rope import make_rope_rotary_cos_sin
from trt.serialize import SerializedTRTEngine
from trt.utils import configure_thor_pytorch

_TEST_ROOT = Path(__file__).resolve().parent.parent
_PLUGIN_CANDIDATES = (
    _TEST_ROOT.parent / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so",
    _TEST_ROOT.parent / "TensorRT-Edge-LLM" / "build-plugin-trt10" / "libNvInfer_edgellm_plugin.so",
)


def ensure_plugin_env() -> None:
    if os.environ.get("EDGE_LLM_PLUGIN_SO") or os.environ.get("EDGELLM_TRT_PLUGIN_SO") or os.environ.get(
        "EDGELLM_PLUGIN_PATH"
    ):
        return
    for candidate in _PLUGIN_CANDIDATES:
        if candidate.is_file():
            os.environ["EDGE_LLM_PLUGIN_SO"] = str(candidate)
            return
    raise RuntimeError(
        "Set EDGE_LLM_PLUGIN_SO to libNvInfer_edgellm_plugin.so "
        f"(looked in: {', '.join(str(p) for p in _PLUGIN_CANDIDATES)})"
    )


def cuda_ms(fn, *, warmup: int = 0) -> tuple[object, float]:
    device = torch.device("cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize(device)
    return result, start.elapsed_time(end)


def denoise_range(
    action_engine,
    actions: torch.Tensor,
    *,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    core,
    start_step: int,
    n_steps: int,
    full_steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Euler updates on the full_steps time grid, running n_steps from start_step."""
    if n_steps <= 0:
        return actions
    x = actions.clone().to(dtype=dtype)
    dt = -1.0 / float(full_steps)
    device = x.device
    batch_size = x.shape[0]
    for step in range(start_step, start_step + n_steps):
        time = 1.0 + step * dt
        timestep = torch.full((batch_size,), time, device=device, dtype=torch.float32)
        pos, mask = make_pi05_suffix_position_and_mask(core, prefix_pad_mask, x, device)
        velocity = action_engine(x, timestep, prefix_k, prefix_v, pos, mask)
        x = x + dt * velocity.to(dtype=dtype)
    return x


def camera_feature_keys(n_cameras: int) -> list[str]:
    keys = [f"{OBS_IMAGES}.image"]
    for index in range(2, n_cameras + 1):
        keys.append(f"{OBS_IMAGES}.image{index}")
    return keys


def align_policy_to_engines(policy, engine_dir: Path) -> None:
    vis = json.loads((engine_dir / "visual" / "config.json").read_text())
    act = json.loads((engine_dir / "action" / "config.json").read_text())
    n_cameras = int(vis["inputs"]["pixel_values"]["shape"][0])
    chunk_size = int(act.get("action_horizon", act["inputs"]["x_t"]["shape"][1]))
    action_dim = int(act.get("action_dim", act["inputs"]["x_t"]["shape"][2]))

    config = policy.config
    config.device = "cpu"
    config.chunk_size = chunk_size
    config.n_action_steps = chunk_size
    config.max_state_dim = action_dim
    config.max_action_dim = action_dim
    config.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for key in camera_feature_keys(n_cameras)
    }
    config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE, shape=(action_dim,)
    )
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    config.empty_cameras = 0
    config.validate_features()


def load_policy(model_id: str, engine_dir: Path):
    policy = PI05Policy.from_pretrained(model_id).cpu().eval()
    align_policy_to_engines(policy, engine_dir)
    model = policy.model.cpu().eval()
    pre_processor, _ = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, model, pre_processor


def prepare_frame(policy, pre_processor, dataset_id: str, episode_index: int, frame_index: int, device, dtype):
    data = load_test_data(dataset_id, episode_index=episode_index, frame_index=frame_index)
    frame = frame_from_test_data(data, policy, fill_missing=True)
    model_inputs = pre_processor(frame)
    images, img_masks = policy._preprocess_images(model_inputs)
    tokens = model_inputs[OBS_LANGUAGE_TOKENS].to(device="cpu", dtype=torch.long)
    masks = model_inputs[OBS_LANGUAGE_ATTENTION_MASK].to(device="cpu", dtype=torch.bool)
    pixel_values = torch.cat(
        [img.to(device=device, dtype=dtype) for img in images],
        dim=0,
    ).contiguous()
    return images, img_masks, tokens, masks, pixel_values


def run_language(
    language: SerializedPi05Language,
    model,
    images,
    img_masks,
    tokens,
    masks,
    image_embs,
    device,
    dtype,
):
    compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids = build_pi05_prefix_embs(
        model,
        img_masks,
        tokens.to(device="cpu"),
        masks.to(device="cpu"),
        image_embs.to(device="cpu"),
        images,
    )
    max_seq_len = int(language.max_seq_len)
    inputs_embeds, prefix_pad_mask, _, prefix_position_ids, valid_seq_len = pad_pi05_compact_prefix(
        compact_embs,
        compact_pad_mask,
        compact_attention_mask,
        compact_position_ids,
        max_seq_len=max_seq_len,
    )
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    prefix_pad_mask = prefix_pad_mask.to(device=device)
    prefix_position_ids = prefix_position_ids.to(device=device)

    paligemma = model.paligemma_with_expert.paligemma.model
    lm = paligemma.language_model
    cfg = lm.config
    bsz, seq_len, hidden = inputs_embeds.shape

    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        seq_len,
        device,
        language_model=lm,
        position_ids=prefix_position_ids,
    )
    ctx_len = torch.full((bsz,), valid_seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), valid_seq_len - 1, device=device, dtype=torch.int64)
    kv_caches = create_kv_caches(cfg, seq_len, bsz, device, dtype=dtype)
    kvcache_start_index = torch.empty(0, device=device, dtype=torch.int32)
    ds_stack = torch.zeros(0, bsz, seq_len, hidden, device=device, dtype=dtype)

    logits, lm_hidden, prefix_k, prefix_v = language(
        inputs_embeds,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        ds_stack,
        *kv_caches,
    )
    return logits, lm_hidden, prefix_k, prefix_v, prefix_pad_mask


def pad_prefix_kv(prefix_k, prefix_v, prefix_pad_mask, prefix_seq_len: int):
    kv_pad = int(prefix_seq_len) - int(prefix_k.shape[-2])
    if kv_pad > 0:
        prefix_k = torch.nn.functional.pad(prefix_k, (0, 0, 0, kv_pad)).contiguous()
        prefix_v = torch.nn.functional.pad(prefix_v, (0, 0, 0, kv_pad)).contiguous()
        prefix_pad_mask = torch.cat(
            [
                prefix_pad_mask,
                torch.zeros(
                    prefix_pad_mask.shape[0],
                    kv_pad,
                    dtype=prefix_pad_mask.dtype,
                    device=prefix_pad_mask.device,
                ),
            ],
            dim=1,
        )
    elif kv_pad < 0:
        prefix_k = prefix_k[..., :prefix_seq_len, :].contiguous()
        prefix_v = prefix_v[..., :prefix_seq_len, :].contiguous()
        prefix_pad_mask = prefix_pad_mask[:, :prefix_seq_len]
    return prefix_k, prefix_v, prefix_pad_mask


def load_pi05_engines(engine_dir: Path):
    missing = [sub for sub in ("visual", "language", "action") if not (engine_dir / sub).is_dir()]
    if missing:
        present = sorted(p.name for p in engine_dir.iterdir()) if engine_dir.is_dir() else []
        raise FileNotFoundError(
            f"{engine_dir} is not a full Pi0.5 export (missing {missing}; found {present}). "
            "Need visual/, language/, and action/ from:\n"
            f"  ENGINE_DIR={engine_dir} python vla/test_vla_pi05_e2e.py"
        )
    vision = SerializedPi05Vision(SerializedTRTEngine(engine_dir / "visual"))
    language = SerializedPi05Language(SerializedTRTEngine(engine_dir / "language"))
    action = SerializedPi05Action(SerializedTRTEngine(engine_dir / "action"))
    return vision, language, action


def setup_cuda():
    configure_thor_pytorch()
    ensure_plugin_env()
    load_plugins_for_trt()
    if not torch.cuda.is_available():
        raise RuntimeError("Pi0.5 TRT inference requires CUDA")
    return torch.device("cuda"), torch.float16
