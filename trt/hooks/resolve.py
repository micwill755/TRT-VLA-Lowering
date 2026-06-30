from __future__ import annotations

import importlib
from typing import Any, Callable


def resolve(path: str) -> Callable[..., Any]:
    """Import ``module.path:callable`` and return the callable."""
    mod_name, _, attr = path.partition(":")
    if not mod_name or not attr:
        raise ValueError(f"Hook path must be 'module.path:callable', got {path!r}")
    return getattr(importlib.import_module(mod_name), attr)
