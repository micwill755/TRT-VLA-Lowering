# Test/smolvla_compile_vla_trt.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_tensor,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from trt.compile import compile_trt_module
from trt.data import make_batch
from trt.measure import compute_action_chunk_ade, tensor_error_metrics
from trt.utils import load_policy
from trt.vision import SmolVLAVisualEmbed
from trt.language import SmolVLAPrefixLanguagePrefill
from trt.diffusion import SmolVLAStaticKVDiffusionStep

MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
SEED = 42

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "min_block_size": 1,
    "use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    #"offload_module_to_cpu": True,
}

@torch.no_grad()
def prepare_smolvla_inputs(policy, batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS].to(state.device)
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(state.device)
    return images, img_masks, tokens, masks, state


@torch.no_grad()
def build_smolvla_prefix_inputs(core, images, img_masks, tokens, masks, state, visual_runner=None):
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

        img_emb = core.vlm_with_expert.embed_image(img) if visual_runner is None else visual_runner(img)
        img_emb = img_emb * torch.tensor(
            img_emb.shape[-1] ** 0.5,
            dtype=img_emb.dtype,
            device=img_emb.device,
        )

        bsize, num_img_embs = img_emb.shape[:2]
        img_mask = img_mask[:, None].expand(bsize, num_img_embs)

        embs.append(img_emb)
        pad_masks.append(img_mask)
        att_masks += [0] * num_img_embs

        if core.add_image_special_tokens:
            image_end_token = (
                core.vlm_with_expert.embed_language_tokens(core.image_end_token.to(device=tokens.device))
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

    prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)[None, :]
    prefix_att_masks = prefix_att_masks.expand(prefix_embs.shape[0], -1)

    if core.prefix_length > 0 and prefix_pad_masks.shape[1] < core.prefix_length:
        prefix_embs = pad_tensor(prefix_embs, core.prefix_length, pad_value=0)
        prefix_pad_masks = pad_tensor(prefix_pad_masks, core.prefix_length, pad_value=0)
        prefix_att_masks = pad_tensor(prefix_att_masks, core.prefix_length, pad_value=0)

    prefix_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    return prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids


def make_suffix_position_and_mask(prefix_pad_masks, x_t, device):
    batch_size, suffix_len = x_t.shape[:2]
    prefix_pad_masks = prefix_pad_masks.to(device=device)
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    attention_mask = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask


def make_compile_inputs(core, action_module, prefix_pad_masks, prefix_k, prefix_v, device):
    dtype = next(action_module.action_in_proj.parameters()).dtype
    batch_size = prefix_pad_masks.shape[0]

    x_t = torch.randn(
        batch_size,
        core.config.chunk_size,
        core.config.max_action_dim,
        device=device,
        dtype=dtype,
    )
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    position_ids, attention_mask = make_suffix_position_and_mask(prefix_pad_masks, x_t, device)

    return (
        x_t,
        timestep,
        prefix_k.to(device=device),
        prefix_v.to(device=device),
        position_ids,
        attention_mask,
    )


def make_runner_inputs(core, prefix_pad_masks, prefix_k, prefix_v, x_t, timestep, device):
    dtype = next(core.action_in_proj.parameters()).dtype
    position_ids, attention_mask = make_suffix_position_and_mask(prefix_pad_masks, x_t, device)

    return (
        x_t.to(device=device, dtype=dtype),
        timestep.to(device=device),
        prefix_k.to(device=device),
        prefix_v.to(device=device),
        position_ids,
        attention_mask,
    )


@torch.no_grad()
def sample_actions_with_full_smolvla_trt(
    policy,
    batch,
    visual_runner,
    language_runner,
    action_runner,
    noise,
    device,
):
    core = policy.model
    images, img_masks, tokens, masks, state = prepare_smolvla_inputs(policy, batch)

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_smolvla_prefix_inputs(
        core,
        images,
        img_masks,
        tokens,
        masks,
        state,
        visual_runner=visual_runner,
    )

    prefix_k, prefix_v = language_runner(prefix_embs, prefix_attention_mask, prefix_position_ids)

    x_t = noise.clone()
    dt = -1.0 / core.config.num_steps

    for step in range(core.config.num_steps):
        timestep = torch.full(
            (x_t.shape[0],),
            1.0 + step * dt,
            dtype=torch.float32,
            device=device,
        )

        v_t = action_runner(
            *make_runner_inputs(core, prefix_pad_masks, prefix_k, prefix_v, x_t, timestep, device)
        ).float()

        x_t = x_t + dt * v_t

    return x_t


@torch.no_grad()
def compare_vision(core, images, visual_runner):
    for i, image in enumerate(images):
        eager = core.vlm_with_expert.embed_image(image)
        trt = visual_runner(image)
        tensor_error_metrics(f"vision[{i}]", trt, eager)


@torch.no_grad()
def compare_language(language_runner, prefix_embs, attention_mask, position_ids):
    eager_k, eager_v = SmolVLAPrefixLanguagePrefill(core).eval().to(prefix_embs.device)(
        prefix_embs,
        attention_mask,
        position_ids,
    )
    trt_k, trt_v = language_runner(prefix_embs, attention_mask, position_ids)

    tensor_error_metrics("language prefix_k", trt_k, eager_k)
    tensor_error_metrics("language prefix_v", trt_v, eager_v)


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_policy(SmolVLAPolicy, MODEL_ID, device, True).eval()
    policy = policy.to(device=device, dtype=torch.float32).eval()

    batch = make_batch(
        policy,
        MODEL_ID,
        device,
        fill_missing=True,
        dataset_id=DATASET_ID,
    )

    core = policy.model.to(device=device, dtype=torch.float32).eval()

    images, img_masks, tokens, masks, state = prepare_smolvla_inputs(policy, batch)
    images = [img.to(device=device, dtype=torch.float32) for img in images]
    img_masks = [mask.to(device=device) for mask in img_masks]
    tokens = tokens.to(device=device)
    masks = masks.to(device=device)
    state = state.to(device=device, dtype=torch.float32)

    print("vision dtype:", next(core.vlm_with_expert.get_vlm_model().vision_model.parameters()).dtype)
    print("action dtype:", next(core.action_in_proj.parameters()).dtype)
    print("image dtype:", images[0].dtype)

    print("compiling vision")
    trt_visual = compile_trt_module(
        SmolVLAVisualEmbed(core).eval().to(device),
        (images[0],),
        TRT_SETTINGS,
    )

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_smolvla_prefix_inputs(
        core,
        images,
        img_masks,
        tokens,
        masks,
        state,
        visual_runner=trt_visual,
    )

    print("compiling language")
    trt_language = compile_trt_module(
        SmolVLAPrefixLanguagePrefill(core).eval().to(device),
        (prefix_embs, prefix_attention_mask, prefix_position_ids),
        TRT_SETTINGS,
    )

    print("metrics")
    compare_vision(core, images, trt_visual)

    eager_prefix_embs, _, eager_prefix_attention_mask, eager_prefix_position_ids = build_smolvla_prefix_inputs(
        core,
        images,
        img_masks,
        tokens,
        masks,
        state,
        visual_runner=None,
    )

    eager_k, eager_v = SmolVLAPrefixLanguagePrefill(core).eval().to(device)(
        eager_prefix_embs,
        eager_prefix_attention_mask,
        eager_prefix_position_ids,
    )
    trt_k, trt_v = trt_language(
        eager_prefix_embs,
        eager_prefix_attention_mask,
        eager_prefix_position_ids,
    )
    tensor_error_metrics("language prefix_k", trt_k, eager_k)
    tensor_error_metrics("language prefix_v", trt_v, eager_v)

    prefix_k, prefix_v = trt_language(
        prefix_embs,
        prefix_attention_mask,
        prefix_position_ids,
    )

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

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

    print("compiling action")
    action_module = SmolVLAStaticKVDiffusionStep(core).eval().to(device)

    sample_inputs = make_compile_inputs(
        core,
        action_module,
        prefix_pad_masks,
        prefix_k,
        prefix_v,
        device,
    )

    trt_action = compile_trt_module(
        action_module,
        sample_inputs,
        TRT_SETTINGS,
    )

    print("full action metrics")
    trt_actions = sample_actions_with_full_smolvla_trt(
        policy,
        batch,
        trt_visual,
        trt_language,
        trt_action,
        noise=noise.clone(),
        device=device,
    )

    action_dim = policy.config.action_feature.shape[0]
    eager_actions = eager_actions[:, :, :action_dim]
    trt_actions = trt_actions[:, :, :action_dim]

    diff = (eager_actions.float() - trt_actions.float()).abs()
    ade = compute_action_chunk_ade(trt_actions, eager_actions)

    print("action xyz ADE:", ade)
    print("action xyz minADE:", ade)
    print("Eager actions:", eager_actions.shape, eager_actions.dtype)
    print("TRT actions:", trt_actions.shape, trt_actions.dtype)
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())