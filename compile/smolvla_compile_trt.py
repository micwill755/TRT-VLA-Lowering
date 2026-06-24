"""In-memory TensorRT compile + parity smoke test for SmolVLA."""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn as nn

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks, pad_tensor
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from trt.action_rollout import ActionRolloutContext, PrefixKVFlowActionAdapter, sample_actions_raw
from trt.compile import compile_trt_module
from trt.data import DEFAULT_DATASET_ID, load_test_data, prepare_policy_batch
from trt.diffusion import SmolVLAPrefixKVStepEncoder, StaticActionVelocityStep
from trt.measure import (
    compare_action_rollout_to_eager,
    compute_action_chunk_ade,
    compute_action_chunk_minade,
    tensor_error_metrics,
)
from trt.plugin_utils import (
    infer_smolvlm_seq_len,
    load_plugin,
    patch_vision_attention,
    restore_attention,
)
from trt.utils import load_policy, make_smolvla_runner_inputs

MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
SEED = 42

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}


class SmolVLAVisualEmbed(nn.Module):
    """Image -> SmolVLM connector output for TRT export."""

    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.vlm_with_expert.embed_image(image)


def stack_smolvla_prefix_kv(past_key_values, num_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert SmolVLM dict cache to stacked [L, B, H, S, D] prefix tensors."""
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_idx in range(int(num_layers)):
        entry = past_key_values[layer_idx]
        key_states = entry["key_states"]
        value_states = entry["value_states"]
        if key_states.ndim != 4:
            raise ValueError(
                f"Expected SmolVLA KV states with shape [B, S, H, D], got {tuple(key_states.shape)}"
            )
        keys.append(key_states.transpose(1, 2).contiguous())
        values.append(value_states.transpose(1, 2).contiguous())
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


class SmolVLAPrefixLanguagePrefill(nn.Module):
    """Prefix-only SmolVLM+expert forward that returns stacked prefix K/V."""

    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert
        self.config = core.config
        self.num_layers = int(core.vlm_with_expert.num_vlm_layers)

    def forward(
        self,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        return stack_smolvla_prefix_kv(past_key_values, self.num_layers)


class _SmolVLAActionExpert(nn.Module):
    """Thin wrapper so StaticActionVelocityStep can call vlm_with_expert.forward."""

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
    )


@torch.no_grad()
def prepare_smolvla_inputs(policy, batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS].to(state.device)
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(state.device)
    return images, img_masks, tokens, masks, state


@torch.no_grad()
def build_smolvla_prefix_inputs(
    core,
    images,
    img_masks,
    tokens,
    masks,
    state,
    *,
    visual_runner=None,
):
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

        img_emb = (
            core.vlm_with_expert.embed_image(img)
            if visual_runner is None
            else visual_runner(img)
        )
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
    ]
    prefix_att_masks = prefix_att_masks.expand(prefix_embs.shape[0], -1)

    if core.prefix_length > 0 and prefix_pad_masks.shape[1] < core.prefix_length:
        prefix_embs = pad_tensor(prefix_embs, core.prefix_length, pad_value=0)
        prefix_pad_masks = pad_tensor(prefix_pad_masks, core.prefix_length, pad_value=0)
        prefix_att_masks = pad_tensor(prefix_att_masks, core.prefix_length, pad_value=0)

    prefix_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    return prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids


def make_compile_inputs(core, action_module, prefix_pad_masks, prefix_k, prefix_v, device):
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
    )


def _smolvla_action_adapter(core) -> PrefixKVFlowActionAdapter:
    return PrefixKVFlowActionAdapter(
        core,
        int(core.config.num_steps),
        runner_inputs_fn=make_smolvla_runner_inputs,
    )


@torch.no_grad()
def compare_smolvla_vision(core, images, visual_runner):
    for i, image in enumerate(images):
        eager = core.vlm_with_expert.embed_image(image)
        trt = visual_runner(image)
        tensor_error_metrics(f"vision[{i}]", trt, eager)


@torch.no_grad()
def compare_smolvla_language(language_runner, prefix_embs, attention_mask, position_ids, core):
    eager_runner = SmolVLAPrefixLanguagePrefill(core).eval().to(prefix_embs.device)
    eager_k, eager_v = eager_runner(prefix_embs, attention_mask, position_ids)
    trt_k, trt_v = language_runner(prefix_embs, attention_mask, position_ids)
    tensor_error_metrics("language prefix_k", trt_k, eager_k)
    tensor_error_metrics("language prefix_v", trt_v, eager_v)


@torch.no_grad()
def compare_smolvla_action_step(
    core,
    action_module,
    action_runner,
    prefix_pad_masks,
    prefix_k,
    prefix_v,
    noise,
    device,
):
    action_module = action_module.to(device).eval()
    timestep = torch.ones(noise.shape[0], dtype=torch.float32, device=device)
    inputs = make_smolvla_runner_inputs(core, prefix_pad_masks, prefix_k, prefix_v, noise, timestep, device)
    eager = action_module(*inputs).float()
    trt = action_runner(*inputs).float()
    tensor_error_metrics("action step output", trt, eager)
    print("action step xyz ADE:", compute_action_chunk_ade(trt, eager))
    print("action step xyz minADE:", compute_action_chunk_minade(trt, eager))


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_policy(SmolVLAPolicy, MODEL_ID, device).eval()
    policy = policy.to(device=device, dtype=torch.float32).eval()

    data = load_test_data(dataset_id=DATASET_ID or DEFAULT_DATASET_ID)
    batch = prepare_policy_batch(
        policy,
        data,
        device,
        MODEL_ID,
        fill_missing=True,
    )

    core = policy.model.to(device=device, dtype=torch.float32).eval()

    images, img_masks, tokens, masks, state = prepare_smolvla_inputs(policy, batch)
    images = [img.to(device=device, dtype=torch.float32) for img in images]
    img_masks = [mask.to(device=device) for mask in img_masks]
    tokens = tokens.to(device=device)
    masks = masks.to(device=device)
    state = state.to(device=device, dtype=torch.float32)

    load_plugin()

    print("compiling vision")
    vision_model = core.vlm_with_expert.get_vlm_model().vision_model
    batch_size, seq_len = infer_smolvlm_seq_len(vision_model, images[0])
    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SmolVLM",
    )
    try:
        trt_visual = compile_trt_module(
            SmolVLAVisualEmbed(core).eval().to(device),
            (images[0],),
            TRT_SETTINGS,
        )
    finally:
        restore_attention(patched)

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
    language_module = SmolVLAPrefixLanguagePrefill(core).eval().to(device)
    trt_language = compile_trt_module(
        language_module,
        (prefix_embs, prefix_attention_mask, prefix_position_ids),
        TRT_SETTINGS,
    )

    print("metrics")
    compare_smolvla_vision(core, images, trt_visual)
    compare_smolvla_language(
        trt_language,
        prefix_embs,
        prefix_attention_mask,
        prefix_position_ids,
        core,
    )

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
    action_module = make_smolvla_action_step(core).eval().to(device)
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
        ACTION_TRT_SETTINGS,
    )

    print("direct action step metrics")
    compare_smolvla_action_step(
        core,
        action_module,
        trt_action,
        prefix_pad_masks,
        prefix_k,
        prefix_v,
        noise,
        device,
    )

    print("full action metrics")
    action_context = ActionRolloutContext(
        noise=noise.clone(),
        device=device,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        prefix_pad_mask=prefix_pad_masks,
    )
    trt_actions = sample_actions_raw(
        trt_action,
        action_context,
        _smolvla_action_adapter(core),
    )

    action_dim = policy.config.action_feature.shape[0]
    compare_action_rollout_to_eager(
        eager_actions[:, :, :action_dim],
        trt_actions[:, :, :action_dim],
        name="full smolvla rollout",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
