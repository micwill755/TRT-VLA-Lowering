import torch
import torch_tensorrt

from trt.diffusion import PI05StaticKVDiffusionStep

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"

MODEL_ID = "lerobot/pi05_libero"
FUTURE_STEPS = 5
SEED = 42

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "min_block_size": 1,
    "use_python_runtime": True,
    "decompose_attention": True,
    "offload_module_to_cpu": True,
}

@torch.no_grad()
def build_prefix(core, images, img_masks, tokens, masks):
    # it computes the KV cache for the observation/task prefix so every diffusion denoising step can reuse it..

    prefix_embs, prefix_pad_masks, prefix_att_masks = core.embed_prefix(
        images, img_masks, tokens, masks
    )

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_attention_mask = core._prepare_attention_masks_4d(prefix_att_2d_masks)

    core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, past_key_values = core.paligemma_with_expert.forward(
        attention_mask=prefix_attention_mask,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    return prefix_pad_masks, past_key_values

def stack_prefix_kv(past_key_values):
    if hasattr(past_key_values, "layers"):
        prefix_k = torch.stack([layer.keys for layer in past_key_values.layers], dim=0)
        prefix_v = torch.stack([layer.values for layer in past_key_values.layers], dim=0)
        return prefix_k, prefix_v

    prefix_k = torch.stack([layer[0] for layer in past_key_values], dim=0)
    prefix_v = torch.stack([layer[1] for layer in past_key_values], dim=0)
    return prefix_k, prefix_v

def make_suffix_position_and_mask(core, prefix_pad_masks, x_t, device):
    batch_size, suffix_len = x_t.shape[:2]
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + [0] * (suffix_len - 1),
        dtype=torch.int64,
        device=device,
    )[None, :].expand(batch_size, -1)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size, suffix_len, prefix_len
    )
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    attention_mask = core._prepare_attention_masks_4d(full_att_2d_masks)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask


def make_runner_inputs(core, prefix_pad_masks, past_key_values, x_t, timestep, device):
    action_dtype = next(core.action_in_proj.parameters()).dtype
    prefix_k, prefix_v = stack_prefix_kv(past_key_values)
    position_ids, attention_mask = make_suffix_position_and_mask(core, prefix_pad_masks, x_t, device)

    return (
        x_t.to(action_dtype),
        timestep,
        prefix_k.to(action_dtype),
        prefix_v.to(action_dtype),
        position_ids,
        attention_mask,
    )

@torch.no_grad()
def sample_actions_with_runner(policy, batch, action_runner, noise, num_steps, device):
    core = policy.model
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch)
    prefix_pad_masks, past_key_values = build_prefix(core, images, img_masks, tokens, masks)

    batch_size = tokens.shape[0]
    dt = -1.0 / num_steps
    x_t = noise.clone()

    for step in range(num_steps):
        timestep = torch.full(
            (batch_size,),
            1.0 + step * dt,
            dtype=torch.float32,
            device=x_t.device,
        )
        v_t = action_runner(
            *make_runner_inputs(core, prefix_pad_masks, past_key_values, x_t, timestep, device)
        ).float()
        x_t = x_t + dt * v_t

    return x_t

@torch.no_grad()
def compare_to_eager(policy, batch, action_runner, device=None, seed=SEED):
    core = policy.model
    num_steps = core.config.num_inference_steps

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch)
    noise = core.sample_noise(
        (tokens.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )

    eager_actions = core.sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        noise=noise.clone(),
        num_steps=num_steps,
    )
    runner_actions = sample_actions_with_runner(
        policy,
        batch,
        action_runner,
        noise=noise.clone(),
        num_steps=num_steps,
    )

    action_dim = policy.config.output_features[ACTION].shape[0]
    eager_actions = eager_actions[:, :, :action_dim]
    runner_actions = runner_actions[:, :, :action_dim]
    diff = (eager_actions.float() - runner_actions.float()).abs()

    ade = compute_action_chunk_ade(runner_actions, eager_actions)
    print("action xyz ADE:", ade)
    print("action xyz minADE:", ade)
    print("Eager actions:", eager_actions.shape, eager_actions.dtype)
    print("Runner actions:", runner_actions.shape, runner_actions.dtype)
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(MODEL_ID, device)
    batch = make_batch(policy, MODEL_ID, device)
    core = policy.model

    action_step = build_action_step(core, device)
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch)
    prefix_pad_masks, _ = build_prefix(core, images, img_masks, tokens, masks)
    sample_inputs = make_compile_inputs(
        core,
        action_step,
        batch_size=tokens.shape[0],
        prefix_len=prefix_pad_masks.shape[1],
        device=device,
    )

    print("compile prefix_len:", prefix_pad_masks.shape[1])

    trt_action_step = compile_action_step(action_step, sample_inputs)
    compare_to_eager(policy, batch, trt_action_step, device=device)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
