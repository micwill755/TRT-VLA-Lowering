from __future__ import annotations

import math

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
    text = prompt.rstrip(".")

    if is_image:
        text = f"{text}. This image is of {height}x{width} resolution."
        system_prompt = _SYSTEM_PROMPT_IMAGE
    else:
        text = f"{text}. The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS."
        text = f"{text}. This video is of {height}x{width} resolution."
        system_prompt = _SYSTEM_PROMPT_VIDEO

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
