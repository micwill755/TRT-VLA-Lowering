from __future__ import annotations

from contextlib import contextmanager

import torch.nn as nn

from exporter import apply_tower_wrappers, register_model_patch, source_policy


def root(policy: nn.Module) -> nn.Module | None:
    if hasattr(policy, "paligemma_with_expert"):
        return policy
    inner = getattr(policy, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "paligemma_with_expert"):
        return inner
    return None


def _match(module: nn.Module) -> bool:
    if not hasattr(module, "vision_tower") or not hasattr(module, "language_model"):
        return False
    return root(source_policy(module)) is not None


@register_model_patch(match=_match)
@contextmanager
def apply_wrappers(module: nn.Module, inputs):
    from .language import wrap_language
    from .vision import wrap_vision

    with apply_tower_wrappers(
        module, inputs, wrap_vision=wrap_vision, wrap_language=wrap_language
    ):
        yield
