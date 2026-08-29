"""Spec-VLA distance-sensitive relaxed acceptance, mapped onto continuous Pi0.5 actions.

Spec-VLA (https://github.com/PineTreeWss/SpecVLA) verifies AR action *tokens* and
accepts a draft token if its bin index is within ``tau`` of the target. Pi0.5 emits
a continuous chunk ``[T, D]`` from flow-matching, so we quantize each dim to
``n_bins`` and apply the same relative-distance test.
"""

from __future__ import annotations

import torch


def quantize_actions(
    actions: torch.Tensor,
    n_bins: int = 256,
    lo: float = -1.0,
    hi: float = 1.0,
) -> torch.Tensor:
    x = actions.float().clamp(lo, hi)
    scale = float(n_bins - 1) / max(hi - lo, 1e-8)
    return ((x - lo) * scale).round().long()


def token_distance(
    draft: torch.Tensor,
    verified: torch.Tensor,
    n_bins: int = 256,
) -> torch.Tensor:
    return (quantize_actions(draft, n_bins) - quantize_actions(verified, n_bins)).abs()


def rmse(draft: torch.Tensor, verified: torch.Tensor) -> float:
    return float((draft.float() - verified.float()).pow(2).mean().sqrt())


def relaxed_accept(
    draft: torch.Tensor,
    verified: torch.Tensor,
    *,
    tau: int,
    n_bins: int = 256,
    gripper_dim: int | None = None,
) -> bool:
    """True if every quantized action dim is within ``tau`` bins of the verify step.

    Gripper (if set) must match exactly, matching Spec-VLA's precision control
    on the discrete gripper token.
    """
    dist = token_distance(draft, verified, n_bins)
    if gripper_dim is not None:
        grip = dist[..., gripper_dim]
        if bool((grip > 0).any()):
            return False
    return bool((dist <= int(tau)).all())


def accept_length(
    draft: torch.Tensor,
    verified: torch.Tensor,
    *,
    tau: int,
    n_bins: int = 256,
) -> int:
    """Count of leading timesteps whose max per-dim bin distance is <= tau."""
    dist = token_distance(draft, verified, n_bins)
    per_t = dist.amax(dim=-1).reshape(-1)
    length = 0
    limit = int(tau)
    for value in per_t.tolist():
        if int(value) <= limit:
            length += 1
        else:
            break
    return length
