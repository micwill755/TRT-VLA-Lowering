import torch

def fused_metric(
    chunk: torch.Tensor,
    xyz_dims=(0, 1, 2),
    alpha: float = 0.5,
    d_scale: float = 1.0,
) -> float:
    """chunk: [T, D] last executed actions. High F = fast/straight, low F = slow/curved."""
    pts = chunk[:, list(xyz_dims)].float()
    if pts.shape[0] < 2:
        return 0.0

    step = (pts[1:] - pts[:-1]).norm(dim=-1)
    path = float(step.sum().clamp_min(1e-8))
    net = float((pts[-1] - pts[0]).norm())
    straight = net / path
    motion = min(path / max(d_scale, 1e-8), 1.0)
    return alpha * straight + (1.0 - alpha) * motion
