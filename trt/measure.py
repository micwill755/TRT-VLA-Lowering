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
    compact_prefix=False,
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

    if compact_prefix:
        from trt.language import compact_prefix_inputs

        prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = compact_prefix_inputs(
            prefix_embs,
            prefix_pad_masks,
            prefix_position_ids,
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


def _first_bad_index(mask, shape):
    flat_idx = int(mask.flatten().nonzero(as_tuple=False)[0].item())
    coords = []
    for dim in reversed(shape):
        coords.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(coords))


def tensor_health_report(name, tensor):
    finite = torch.isfinite(tensor)
    bad = ~finite
    bad_count = int(bad.sum().item())
    if bad_count == 0:
        return

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    first_idx = _first_bad_index(bad, tensor.shape)
    first_val = tensor[first_idx].detach().cpu().item()
    print(f"{name} nonfinite count:", bad_count, "of", tensor.numel())
    print(f"{name} nan count:", nan_count)
    print(f"{name} inf count:", inf_count)
    print(f"{name} first nonfinite index:", first_idx, "value:", first_val)

def tensor_error_metrics(name, trt, eager):
    tensor_health_report(f"{name} TRT", trt)
    tensor_health_report(f"{name} eager", eager)
    trt = trt.float()
    eager = eager.float()
    diff = (trt - eager).abs()

    rel_l2 = (trt - eager).norm() / eager.norm().clamp_min(1e-8)
    rel_mean_pct = diff.mean() / eager.abs().mean().clamp_min(1e-8) * 100

    print(f"{name} mean diff:", diff.mean().item())
    print(f"{name} max diff:", diff.max().item())
    print(f"{name} relative L2:", rel_l2.item())
    print(f"{name} relative mean %:", rel_mean_pct.item())

def _select_hidden_valid(x, valid):
    valid = valid.to(device=x.device, dtype=torch.bool)
    return torch.cat([x[b, valid[b], :] for b in range(valid.shape[0])], dim=0)

def _select_prefix_kv_valid(x, valid):
    valid = valid.to(device=x.device, dtype=torch.bool)
    chunks = [x[:, b, :, valid[b], :].reshape(-1, x.shape[-1]) for b in range(valid.shape[0])]
    return torch.cat(chunks, dim=0)

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
    device,
    compact_prefix=False,
):
    trt_actions = sample_actions_with_full_trt(
        policy,
        batch,
        visual_runner,
        prefix_runner,
        action_runner,
        noise=noise.clone(),
        num_steps=num_steps,
        device=device,
        compact_prefix=compact_prefix,
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
def compare_action_step(core, action_module, action_runner, prefix_pad_masks, prefix_k, prefix_v, x_t, timestep, device):
    action_module = action_module.to(device).eval()
    inputs = make_runner_inputs(core, prefix_pad_masks, prefix_k, prefix_v, x_t, timestep, device)
    eager = action_module(*inputs).float()
    trt = action_runner(*inputs).float()

    tensor_error_metrics("action step output", trt, eager)
    print("action step xyz ADE:", compute_action_chunk_ade(trt, eager))
    print("action step xyz minADE:", compute_action_chunk_minade(trt, eager))

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
    return out.last_hidden_state, prefix_k, prefix_v

@torch.no_grad()
def compare_vision(core, images, visual_runner, eager_runner=None):
    for i, img in enumerate(images):
        eager = eager_runner(img) if eager_runner is not None else core.paligemma_with_expert.embed_image(img)
        trt = visual_runner(img)
        tensor_error_metrics(f"vision[{i}]", trt, eager)

@torch.no_grad()
def compare_language(language_model, prefix_runner, prefix_embs, attention_mask, position_ids, prefix_pad_masks=None):
    eager_hidden, eager_k, eager_v = run_prefix_language_eager(
        language_model,
        prefix_embs,
        attention_mask,
        position_ids,
    )
    trt_hidden, trt_k, trt_v = prefix_runner(
        prefix_embs,
        attention_mask,
        position_ids,
        return_hidden=True,
    )

    tensor_error_metrics("language hidden", trt_hidden, eager_hidden)
    tensor_error_metrics("language prefix_k", trt_k, eager_k)
    tensor_error_metrics("language prefix_v", trt_v, eager_v)

    if prefix_pad_masks is not None:
        tensor_error_metrics(
            "language hidden valid",
            _select_hidden_valid(trt_hidden, prefix_pad_masks),
            _select_hidden_valid(eager_hidden, prefix_pad_masks),
        )
        tensor_error_metrics(
            "language prefix_k valid",
            _select_prefix_kv_valid(trt_k, prefix_pad_masks),
            _select_prefix_kv_valid(eager_k, prefix_pad_masks),
        )
        tensor_error_metrics(
            "language prefix_v valid",
            _select_prefix_kv_valid(trt_v, prefix_pad_masks),
            _select_prefix_kv_valid(eager_v, prefix_pad_masks),
        )