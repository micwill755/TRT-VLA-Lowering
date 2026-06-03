import torch
from lerobot.utils.constants import ACTION
from trt.utils import prepare_policy_inputs, make_runner_inputs, build_prefix_inputs

@torch.no_grad()
def sample_actions_with_full_trt(
    policy,
    batch,
    visual_runner,
    prefix_runner,
    action_runner,
    noise,
    num_steps,
    device,
):
    core = policy.model
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch)

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_prefix_inputs(
        core,
        images,
        img_masks,
        tokens,
        masks,
        visual_runner=visual_runner,
    )

    prefix_k, prefix_v = prefix_runner(
        prefix_embs,
        prefix_attention_mask,
        prefix_position_ids,
    )

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
            *make_runner_inputs(core, prefix_pad_masks, prefix_k, prefix_v, x_t, timestep, device)
        ).float()

        x_t = x_t + dt * v_t

    return x_t

def tensor_error_metrics(name, trt, eager):
    trt = trt.float()
    eager = eager.float()
    diff = (trt - eager).abs()

    rel_l2 = (trt - eager).norm() / eager.norm().clamp_min(1e-8)
    rel_mean_pct = diff.mean() / eager.abs().mean().clamp_min(1e-8) * 100

    print(f"{name} mean diff:", diff.mean().item())
    print(f"{name} max diff:", diff.max().item())
    print(f"{name} relative L2:", rel_l2.item())
    print(f"{name} relative mean %:", rel_mean_pct.item())

def compute_action_chunk_ade(pred, target):
    pred_xyz = pred[..., :3].float()
    target_xyz = target[..., :3].float()
    step_l2 = torch.linalg.vector_norm(pred_xyz - target_xyz, dim=-1)
    return step_l2.mean().item()

def compute_action_chunk_minade(pred, target):
    return compute_action_chunk_ade(pred, target)

@torch.no_grad()    
def compare_full_vla_to_eager_actions(
    policy,
    batch,
    visual_runner,
    prefix_runner,
    action_runner,
    eager_actions,
    noise,
    num_steps,
    device
):
    trt_actions = sample_actions_with_full_trt(
        policy,
        batch,
        visual_runner,
        prefix_runner,
        action_runner,
        noise=noise.clone(),
        num_steps=num_steps,
        device=device
    )

    action_dim = policy.config.output_features[ACTION].shape[0]
    eager_actions = eager_actions[:, :, :action_dim]
    trt_actions = trt_actions[:, :, :action_dim]

    diff = (eager_actions.float() - trt_actions.float()).abs()
    ade = compute_action_chunk_ade(trt_actions, eager_actions)
    minade = compute_action_chunk_minade(trt_actions, eager_actions)

    print("action xyz ADE:", ade)
    print("action xyz minADE:", minade)
    print("Eager actions:", eager_actions.shape, eager_actions.dtype)
    print("TRT actions:", trt_actions.shape, trt_actions.dtype)
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())

@torch.no_grad()
def run_prefix_language_eager(language_model, prefix_embs, attention_mask, position_ids):
    out = language_model(
        inputs_embeds=prefix_embs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
    )
    cache = out.past_key_values
    prefix_k = torch.stack([layer.keys for layer in cache.layers], dim=0)
    prefix_v = torch.stack([layer.values for layer in cache.layers], dim=0)
    return prefix_k, prefix_v

@torch.no_grad()
def compare_vision(core, images, visual_runner):
    for i, img in enumerate(images):
        eager = core.paligemma_with_expert.embed_image(img)
        trt = visual_runner(img)
        tensor_error_metrics(f"vision[{i}]", trt, eager)

@torch.no_grad()
def compare_language(language_model, prefix_runner, prefix_embs, attention_mask, position_ids):
    eager_k, eager_v = run_prefix_language_eager(
        language_model,
        prefix_embs,
        attention_mask,
        position_ids,
    )
    trt_k, trt_v = prefix_runner(
        prefix_embs,
        attention_mask,
        position_ids,
    )

    tensor_error_metrics("language prefix_k", trt_k, eager_k)
    tensor_error_metrics("language prefix_v", trt_v, eager_v)