import torch
from lerobot.utils.constants import ACTION
from trt.action_rollout import (
    ActionRolloutContext,
    GROOTActionAdapter,
    PrefixKVFlowActionAdapter,
    sample_actions_raw,
)
from trt.packing import pack_pi05_prefix
from trt.utils import prepare_policy_inputs, make_runner_inputs, compact_prefix_inputs

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

def parity_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Compare ``a`` (reference) against ``b`` and return parity metrics."""
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    rel_l2 = (a - b).norm() / b.norm().clamp_min(1e-8)
    rel_mean_pct = diff.mean() / b.abs().mean().clamp_min(1e-8) * 100
    close = torch.isclose(a, b, rtol=1e-2, atol=1e-2).float().mean() * 100

    return {
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "rel_l2": float(rel_l2.item()),
        "rel_mean_pct": float(rel_mean_pct.item()),
        "close_pct": float(close.item()),
    }


def parity(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Engine parity report (mirrors test_vla.py:612 parity()).

    Compares ``a`` (eager reference) against ``b`` (TRT/serialized) and prints the
    standard row used across stages. Returns the metrics so callers can collect or
    assert on them.
    """
    metrics = parity_metrics(a, b)
    print(
        f"{name:<22} mean_abs={metrics['mean_abs']:.6f}  max_abs={metrics['max_abs']:.6f}  "
        f"rel_l2={metrics['rel_l2']:.4f}  rel_mean%={metrics['rel_mean_pct']:.2f}  "
        f"close%={metrics['close_pct']:.1f}"
    )
    return metrics


def tensor_parity_metrics(trt: torch.Tensor, eager: torch.Tensor) -> dict[str, float]:
    trt = trt.float()
    eager = eager.float()
    diff = (trt - eager).abs()
    rel_l2 = (trt - eager).norm() / eager.norm().clamp_min(1e-8)
    relmean_pct = diff.mean() / eager.abs().mean().clamp_min(1e-8) * 100
    return {
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "relative_l2": float(rel_l2.item()),
        "relative_mean_pct": float(relmean_pct.item()),
    }


def tensor_error_metrics(name: str, trt: torch.Tensor, eager: torch.Tensor) -> dict[str, float]:
    tensor_health_report(f"{name} TRT", trt)
    tensor_health_report(f"{name} eager", eager)
    metrics = tensor_parity_metrics(trt, eager)
    print(f"{name} mean diff:", metrics["mean_abs"])
    print(f"{name} max diff:", metrics["max_abs"])
    print(f"{name} relative L2:", metrics["relative_l2"])
    print(f"{name} relative mean %:", metrics["relative_mean_pct"])
    return metrics


def compare_image_embeddings(
    eager_embs: torch.Tensor,
    trt_embs: torch.Tensor,
    *,
    name: str = "vision image_embs",
) -> dict[str, float]:
    metrics = tensor_parity_metrics(trt_embs, eager_embs)
    print(
        f"  {name:<28} mean_abs={metrics['mean_abs']:.6f}  "
        f"max_abs={metrics['max_abs']:.6f}  rel_l2={metrics['relative_l2']:.6f}"
    )
    return metrics


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

@torch.no_grad()
def compare_action_rollout_to_eager(eager_actions, trt_actions, *, action_dim=None, name=None):
    if action_dim is not None:
        eager_actions = eager_actions[:, :, :action_dim]
        trt_actions = trt_actions[:, :, :action_dim]

    eager_actions = eager_actions.float()
    trt_actions = trt_actions.float()
    diff = (eager_actions - trt_actions).abs()

    if name is not None:
        print(name)

    print("action xyz ADE:", compute_action_chunk_ade(trt_actions, eager_actions))
    print("action xyz minADE:", compute_action_chunk_minade(trt_actions, eager_actions))
    print("Eager actions:", eager_actions.shape, eager_actions.dtype)
    print("TRT actions:", trt_actions.shape, trt_actions.dtype)
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())

@torch.no_grad()
def compare_actions_raw(
    action_runner,
    eager_actions,
    context: ActionRolloutContext,
    adapter,
    *,
    action_dim=None,
    name=None,
):
    trt_actions = sample_actions_raw(action_runner, context, adapter)
    compare_action_rollout_to_eager(
        eager_actions,
        trt_actions,
        action_dim=action_dim,
        name=name,
    )
    return trt_actions

@torch.no_grad()
def compare_groot_action_step(action_module, trt_action, vl_embs, state, embodiment_id, noise, device):
    dtype = vl_embs.dtype
    batch_size = vl_embs.shape[0]

    vl_embs = vl_embs.to(device=device, dtype=dtype)
    actions = noise.clone().to(device=device, dtype=dtype)
    state = state.to(device=device, dtype=dtype)
    embodiment_id = embodiment_id.to(device=device)

    timestep = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.long,
    )

    eager = action_module(
        actions,
        timestep,
        vl_embs,
        state,
        embodiment_id,
    )

    trt = trt_action(
        actions,
        timestep,
        vl_embs,
        state,
        embodiment_id,
    )
    if isinstance(trt, (tuple, list)):
        trt = trt[0]

    tensor_error_metrics("action step output", trt, eager)
    print("action step xyz ADE:", compute_action_chunk_ade(trt, eager))
    print("action step xyz minADE:", compute_action_chunk_minade(trt, eager))


@torch.no_grad()
def compare_full_groot_to_eager_actions(
    core,
    action_runner,
    context_embs,
    state,
    embodiment_id,
    eager_actions,
    noise,
    device,
    *,
    name=None,
    action_dim=None,
):
    context = ActionRolloutContext(
        noise=noise,
        device=device,
        context_embs=context_embs,
        state=state,
        embodiment_id=embodiment_id,
    )
    return compare_actions_raw(
        action_runner,
        eager_actions,
        context,
        GROOTActionAdapter(core.action_head),
        action_dim=action_dim,
        name=name,
    )

@torch.no_grad()    
def compare_full_vla_to_eager_actions(
    policy,
    batch,
    prefix_k, 
    prefix_v,
    visual_runner,
    action_runner,
    eager_actions,
    noise,
    num_steps,
    device,
    compact_prefix=False,
):
    core = policy.model
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)

    image_embs = [visual_runner(image) for image in images]

    packed = pack_pi05_prefix(core, image_embs, img_masks, tokens, masks, compact=False)
    prefix_embs = packed["inputs_embeds"]
    prefix_pad_masks = packed["pad_mask"]
    prefix_attention_mask = packed["attention_mask"]
    prefix_position_ids = packed["position_ids"]

    if compact_prefix:
        prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = compact_prefix_inputs(
            prefix_embs,
            prefix_pad_masks,
            prefix_position_ids,
        )

    context = ActionRolloutContext(
        noise=noise,
        device=device,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        prefix_pad_mask=prefix_pad_masks,
    )
    return compare_actions_raw(
        action_runner,
        eager_actions,
        context,
        PrefixKVFlowActionAdapter(core, num_steps),
        action_dim=policy.config.output_features[ACTION].shape[0],
    )

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
def compare_language(eager_hidden, eager_k, eager_v, trt_hidden, trt_k, trt_v, prefix_pad_masks=None):
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

def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def std(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5

def print_timing(name: str, times_ms: list[float]) -> None:
    if len(times_ms) == 0:
        return
    print(
        f"  {name:<22} min={min(times_ms):7.1f}  avg={mean(times_ms):7.1f}  "
        f"max={max(times_ms):7.1f}  std={std(times_ms):6.1f}  (ms)"
    )

def print_action_metrics(name: str, values: list[float]) -> None:
    if len(values) == 0:
        return
        
    print(
        f"  {name:<22} min={min(values):9.6f}  avg={mean(values):9.6f}  "
        f"max={max(values):9.6f}  std={std(values):9.6f}"
    )

def compute_action_parity_metrics(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> dict[str, float]:
    pred = pred_actions.float()
    target = target_actions.float()

    diff = pred - target
    abs_diff = diff.abs()

    step_l2 = torch.linalg.vector_norm(diff, dim=-1)

    return {
        "action_ade": float(step_l2.mean().item()),
        "action_fde": float(step_l2[..., -1].mean().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
    }
