from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."

def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)

    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    *,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    fps_modulation_enabled = fps is not None and grid_t > 1

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    return mrope_ids, math.ceil(mrope_ids.max().item()) + 1


def format_cosmos_user_prompt(
    prompt: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
) -> str:
    """Augment the user prompt the same way Cosmos training/inference does."""
    is_image = num_frames == 1
    text = prompt.rstrip(".")

    if is_image:
        return f"{text}. This image is of {height}x{width} resolution."
    return (
        f"{text}. The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS."
        f". This video is of {height}x{width} resolution."
    )


def cosmos_wfm_chat_template() -> dict:
    """``processed_chat_template.json`` matching Cosmos Qwen-style HF chat templates."""
    return {
        "roles": {
            "system": {
                "prefix": "\n<|im_start|>system\n",
                "suffix": "<|im_end|>\n",
            },
            "user": {
                "prefix": "<|im_start|>user\n",
                "suffix": "<|im_end|>\n",
            },
            "assistant": {
                "prefix": "<|im_start|>assistant\n",
                "suffix": "<|im_end|>\n",
            },
        },
        "content_types": {},
        "generation_prompt": "<|im_start|>assistant\n<think>\n",
        "default_system_prompt": "",
    }


def save_wfm_tokenizer_assets(engine_root: Path, *, tokenizer: Any) -> None:
    """Write HF tokenizer + ``processed_chat_template.json`` for ``WFMInferenceRuntime``."""
    from trt.tokenizer import save_tokenizer_for_edge_llm

    tokenizer_dir = Path(engine_root) / "tokenizer"
    save_tokenizer_for_edge_llm(
        tokenizer_dir,
        tokenizer=tokenizer,
        chat_template=cosmos_wfm_chat_template(),
    )


def tokenize_cosmos_prompt(
    tokenizer,
    prompt: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
    use_system_prompt: bool = True,
) -> list[int]:
    is_image = num_frames == 1
    text = format_cosmos_user_prompt(
        prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
    )
    system_prompt = _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO

    conversations = []
    if use_system_prompt:
        conversations.append({"role": "system", "content": system_prompt})
    conversations.append({"role": "user", "content": text})

    enc = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=True,
    )

    return list(enc.input_ids) + [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|vision_start|>"),
    ]


def encode_cosmos_video(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels [B,3,T,H,W] -> normalized Cosmos latents."""
    in_dtype = pixels.dtype
    dtype = vae.dtype

    raw_mu = vae.encode(pixels.to(dtype)).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device=raw_mu.device, dtype=dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=raw_mu.device, dtype=dtype)
    return ((raw_mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(in_dtype)


def decode_cosmos_video(vae, latents: torch.Tensor) -> torch.Tensor:
    """normalized Cosmos latents [B,C,T,H,W] -> pixels [B,3,T,H,W]."""
    in_dtype = latents.dtype
    dtype = vae.dtype
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=dtype)
    z_raw = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
    return vae.decode(z_raw).sample.to(in_dtype)


def build_cosmos_packed_static(
    *,
    transformer,
    vae,
    tokenizer,
    device: torch.device | str,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    pixels: torch.Tensor | None = None,
    fps: float = 24.0,
    condition_frame_indexes: tuple[int, ...] = (0,),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    cfg = transformer.config
    device = torch.device(device)

    input_ids_list = tokenize_cosmos_prompt(
        tokenizer,
        prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
    )

    und_len = len(input_ids_list)
    text_mrope_ids, next_offset = get_3d_mrope_ids_text_tokens(
        und_len,
        temporal_offset=0,
        use_float_positions=cfg.enable_fps_modulation,
    )
    vision_start_offset = next_offset + cfg.unified_3d_mrope_temporal_modality_margin

    if pixels is None:
        pixels = torch.randn(1, 3, num_frames, height, width, device=device, dtype=dtype)
    else:
        pixels = pixels.to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = encode_cosmos_video(vae, pixels)

    _, _, latent_t, latent_h, latent_w = latents.shape
    patch = int(cfg.latent_patch_size)
    patch_h = math.ceil(latent_h / patch)
    patch_w = math.ceil(latent_w / patch)
    num_vision_tokens = latent_t * patch_h * patch_w
    curr = und_len

    cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < latent_t}
    noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in cond_frames],
        device=device,
        dtype=torch.long,
    )

    frame_token_stride = patch_h * patch_w
    mse_loss_indexes: list[int] = []
    for frame_idx in noisy_frame_indexes.tolist():
        frame_start = curr + frame_idx * frame_token_stride
        mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=vision_start_offset,
        reset_spatial_indices=cfg.unified_3d_mrope_reset_spatial_ids,
        fps=fps if cfg.enable_fps_modulation else None,
        base_fps=float(cfg.base_fps),
        temporal_compression_factor=int(vae.config.scale_factor_temporal),
    )

    position_ids = torch.cat(
        [text_mrope_ids.to(device), vision_mrope_ids.to(device)],
        dim=-1,
    )

    packed_static = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long, device=device),
        "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
        "und_len": und_len,
        "position_ids": position_ids,
        "sequence_length": und_len + num_vision_tokens,
        "vision_token_shapes": [(latent_t, patch_h, patch_w)],
        "vision_sequence_indexes": torch.arange(
            curr,
            curr + num_vision_tokens,
            dtype=torch.long,
            device=device,
        ),
        "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
        "vision_noisy_frame_indexes": [noisy_frame_indexes],
        "num_noisy_tokens": len(noisy_frame_indexes) * frame_token_stride,
    }
    return packed_static, latents, pixels


def encode_cosmos_sound(sound_tokenizer, waveform: torch.Tensor) -> torch.Tensor:
    """waveform ``[B, C, N]`` -> sound latents ``[C, T]`` (batch item 0)."""
    in_dtype = waveform.dtype
    tokenizer_device = next(sound_tokenizer.parameters()).device
    encoder_dtype = next(sound_tokenizer.parameters()).dtype
    waveform = waveform.to(device=tokenizer_device, dtype=encoder_dtype)
    encoded = sound_tokenizer.encode(waveform, return_dict=True)
    latents = encoded.latent_dist.mode()
    return latents[0].to(in_dtype)


def decode_cosmos_sound(sound_tokenizer, latents: torch.Tensor) -> torch.Tensor:
    """sound latents ``[C, T]`` (or ``[B, C, T]``) -> waveform ``[audio_ch, N]``."""
    if latents.ndim == 3:
        latents = latents[0]
    decoder_dtype = next(sound_tokenizer.parameters()).dtype
    return sound_tokenizer.decode(latents.to(dtype=decoder_dtype))


def _sound_waveform_channels(sound_tokenizer) -> int:
    """Encoder input channels for waveform ``[B, C, N]`` (accounts for stereo doubling)."""
    cfg = sound_tokenizer.config
    input_channels = int(cfg.input_channels)
    stereo = bool(getattr(cfg, "stereo", False))
    return input_channels * (2 if stereo else 1)


def compute_sound_latent_length(
    *,
    num_frames: int,
    fps: float,
    sampling_rate: int,
    hop_size: int,
) -> int:
    n_audio_samples = int(num_frames / fps * sampling_rate)
    return (n_audio_samples + hop_size - 1) // hop_size


def build_cosmos_omni_packed_static(
    *,
    transformer,
    vae,
    sound_tokenizer,
    tokenizer,
    device: torch.device | str,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    pixels: torch.Tensor | None = None,
    waveform: torch.Tensor | None = None,
    fps: float = 24.0,
    condition_frame_indexes: tuple[int, ...] = (0,),
    dtype: torch.dtype = torch.bfloat16,
    enable_sound: bool = True,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Edge vision packing plus optional Omni sound segment metadata."""
    packed_static, latents, pixels = build_cosmos_packed_static(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        pixels=pixels,
        fps=fps,
        condition_frame_indexes=condition_frame_indexes,
        dtype=dtype,
    )

    if not enable_sound:
        return packed_static, latents, pixels, None, None

    cfg = transformer.config
    if not getattr(cfg, "sound_gen", False):
        raise ValueError("Transformer config sound_gen=False; use an Omni-capable checkpoint.")

    device = torch.device(device)
    und_len = int(packed_static["und_len"])
    num_vision_tokens = int(packed_static["vision_sequence_indexes"].numel())

    hop_size = int(sound_tokenizer._hop_size)
    sampling_rate = int(sound_tokenizer.config.sampling_rate)
    sound_len = compute_sound_latent_length(
        num_frames=num_frames,
        fps=fps,
        sampling_rate=sampling_rate,
        hop_size=hop_size,
    )

    tokenizer_device = next(sound_tokenizer.parameters()).device
    if waveform is None:
        n_audio_samples = int(num_frames / fps * sampling_rate)
        audio_channels = _sound_waveform_channels(sound_tokenizer)
        waveform = torch.randn(1, audio_channels, n_audio_samples, device=tokenizer_device, dtype=dtype)
    else:
        waveform = waveform.to(device=tokenizer_device, dtype=dtype)

    clean_sound = encode_cosmos_sound(sound_tokenizer, waveform).to(device)
    waveform = waveform.cpu()

    text_mrope_ids, next_offset = get_3d_mrope_ids_text_tokens(
        und_len,
        temporal_offset=0,
        use_float_positions=cfg.enable_fps_modulation,
    )
    vision_start_offset = next_offset + cfg.unified_3d_mrope_temporal_modality_margin
    sound_fps = float(cfg.sound_latent_fps)
    sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=sound_len,
        grid_h=1,
        grid_w=1,
        temporal_offset=vision_start_offset,
        reset_spatial_indices=cfg.unified_3d_mrope_reset_spatial_ids,
        fps=sound_fps if cfg.enable_fps_modulation else None,
        base_fps=float(cfg.base_fps),
        temporal_compression_factor=1,
    )

    curr = und_len + num_vision_tokens
    sound_sequence_indexes = torch.arange(curr, curr + sound_len, dtype=torch.long, device=device)
    sound_noisy_frame_indexes = torch.arange(sound_len, device=device, dtype=torch.long)

    text_cols = packed_static["position_ids"][:, :und_len]
    vision_cols = packed_static["position_ids"][:, und_len : und_len + num_vision_tokens]
    position_ids = torch.cat([text_cols, vision_cols, sound_mrope_ids.to(device)], dim=1)

    packed_static.update(
        {
            "position_ids": position_ids,
            "sequence_length": und_len + num_vision_tokens + sound_len,
            "sound_token_shapes": [(sound_len, 1, 1)],
            "sound_sequence_indexes": sound_sequence_indexes,
            "sound_mse_loss_indexes": sound_sequence_indexes.clone(),
            "sound_noisy_frame_indexes": [sound_noisy_frame_indexes],
            "sound_len": sound_len,
            "num_noisy_sound_tokens": sound_len,
            "sound_latent_fps": sound_fps,
        }
    )
    return packed_static, latents, pixels, clean_sound, waveform


def _tensor_to_json_list(value) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_tensor_to_json_list(item) for item in value]
    return value


def _resolve_mrope_section(cfg) -> list[int]:
    section = getattr(cfg, "mrope_section", None)
    if section is not None:
        return list(section)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
        return list(rope_scaling["mrope_section"])
    return [16, 24, 24]


def build_wfm_root_config(
    *,
    transformer,
    vae,
    sound_tokenizer=None,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    enable_sound: bool = True,
    num_inference_steps: int = 35,
) -> dict:
    """Root ``config.json`` consumed by ``WFMInferenceRuntime``."""
    cfg = transformer.config
    sample_rate = 48000
    if sound_tokenizer is not None:
        sample_rate = int(sound_tokenizer.config.sampling_rate)
    return {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "fps": fps,
        "hidden_size": int(cfg.hidden_size),
        "num_inference_steps": num_inference_steps,
        "max_batch_size": 1,
        "num_train_timesteps": int(getattr(cfg, "num_train_timesteps", 1000)),
        "flow_shift": 5.0,
        "scheduler_solver_order": 2,
        "head_dim": int(cfg.head_dim),
        "rope_theta": float(cfg.rope_theta),
        "enable_fps_modulation": bool(cfg.enable_fps_modulation),
        "unified_3d_mrope_temporal_modality_margin": int(cfg.unified_3d_mrope_temporal_modality_margin),
        "unified_3d_mrope_reset_spatial_ids": bool(cfg.unified_3d_mrope_reset_spatial_ids),
        "base_fps": float(cfg.base_fps),
        "temporal_compression_factor": int(vae.config.scale_factor_temporal),
        "mrope_section": _resolve_mrope_section(cfg),
        "vision_start_token": "<|vision_start|>",
        "use_chat_template": True,
        "use_system_prompt": True,
        "system_prompt_video": _SYSTEM_PROMPT_VIDEO,
        "enable_sound": enable_sound,
        "sound_dim": int(getattr(cfg, "sound_dim", None) or 64),
        "sample_rate": sample_rate,
        "sound_latent_fps": float(getattr(cfg, "sound_latent_fps", 25.0)),
    }


def serialize_cosmos_packed_static(packed_static: dict) -> dict:
    """JSON-serializable ``packing_static.json`` for the C++ WFM runtime."""
    position_ids = packed_static["position_ids"]
    if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 2:
        flat_position_ids = position_ids.detach().cpu().reshape(-1).tolist()
    else:
        flat_position_ids = _tensor_to_json_list(position_ids)

    vision_noisy = packed_static["vision_noisy_frame_indexes"]
    if isinstance(vision_noisy, (list, tuple)):
        vision_noisy = vision_noisy[0]

    payload = {
        "input_ids": _tensor_to_json_list(packed_static["input_ids"]),
        "text_indexes": _tensor_to_json_list(packed_static["text_indexes"]),
        "und_len": int(packed_static["und_len"]),
        "position_ids": flat_position_ids,
        "sequence_length": int(packed_static["sequence_length"]),
        "vision_token_shapes": packed_static["vision_token_shapes"],
        "vision_sequence_indexes": _tensor_to_json_list(packed_static["vision_sequence_indexes"]),
        "vision_mse_loss_indexes": _tensor_to_json_list(packed_static["vision_mse_loss_indexes"]),
        "vision_noisy_frame_indexes": _tensor_to_json_list(vision_noisy),
        "num_noisy_tokens": int(packed_static["num_noisy_tokens"]),
    }

    if "sound_token_shapes" in packed_static:
        sound_noisy = packed_static["sound_noisy_frame_indexes"]
        if isinstance(sound_noisy, (list, tuple)):
            sound_noisy = sound_noisy[0]
        sound_slots = _tensor_to_json_list(sound_noisy)
        payload.update(
            {
                "sound_token_shapes": packed_static["sound_token_shapes"],
                "sound_sequence_indexes": _tensor_to_json_list(packed_static["sound_sequence_indexes"]),
                "sound_mse_loss_indexes": _tensor_to_json_list(packed_static["sound_mse_loss_indexes"]),
                "sound_noisy_slot_indexes": sound_slots,
                "num_noisy_sound_tokens": int(packed_static.get("num_noisy_sound_tokens", len(sound_slots))),
            }
        )

    return payload
