"""GR00T adapter registrations. Patch bodies live in ``patches/``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from exporter import register_edge_adapter

from .config import SAVE
from .patches import root
from .patches.action import wrap_action, wrap_action_context
from .patches.fuse import fuse_spec
from .patches.language import discover as discover_language
from .patches.vision import discover as discover_vision


def is_groot(policy: nn.Module) -> bool:
    return root(policy) is not None


@register_edge_adapter("vision", match=is_groot)
def vision(policy: nn.Module, _inputs: Mapping[str, Any]) -> nn.Module:
    return discover_vision(policy)


@register_edge_adapter("language", match=is_groot)
def language(policy: nn.Module, _inputs: Mapping[str, Any]) -> nn.Module:
    return discover_language(policy)


@register_edge_adapter("fuse", match=is_groot)
def fuse(_policy: nn.Module, _inputs: Mapping[str, Any]):
    return fuse_spec()


@register_edge_adapter("action_context", match=is_groot)
def action_context(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    return wrap_action_context(policy, inputs)


@register_edge_adapter("action", match=is_groot)
def action(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    return wrap_action(policy, inputs)


@register_edge_adapter("save", match=is_groot)
def save(_policy: nn.Module, _inputs: Mapping[str, Any]):
    return SAVE
