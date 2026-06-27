"""Shared TensorRT compile settings for VLA export."""

from __future__ import annotations

TRT_SETTINGS: dict = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

VISION_TRT_SETTINGS: dict = {
    **TRT_SETTINGS,
    "use_fp32_acc": True,
}

ACTION_TRT_SETTINGS: dict = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
    "use_fp32_acc": True,
}


def in_memory_settings(base: dict) -> dict:
    return {**base, "use_python_runtime": True}
