"""PI05 adapter registrations. Patch bodies live in ``patches/``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from exporter import register_edge_adapter

from .config import SAVE
from .patches import root
from .patches.action import wrap_action
from .patches.fuse import fuse_spec
from .patches.language import discover as discover_language
from .patches.vision import discover as discover_vision


def is_pi05(policy: nn.Module) -> bool:
    return root(policy) is not None


@register_edge_adapter("vision", match=is_pi05)
def vision(policy: nn.Module, _inputs: Mapping[str, Any]) -> nn.Module:
    return discover_vision(policy)


@register_edge_adapter("language", match=is_pi05)
def language(policy: nn.Module, _inputs: Mapping[str, Any]) -> nn.Module:
    return discover_language(policy)


@register_edge_adapter("fuse", match=is_pi05)
def fuse(_policy: nn.Module, _inputs: Mapping[str, Any]):
    return fuse_spec()


@register_edge_adapter("action", match=is_pi05)
def action(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    return wrap_action(policy, inputs)


@register_edge_adapter("save", match=is_pi05)
def save(_policy: nn.Module, _inputs: Mapping[str, Any]):
    return SAVE
