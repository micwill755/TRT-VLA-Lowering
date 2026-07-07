import torch


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


def compute_action_parity_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> dict[str, float]:
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
