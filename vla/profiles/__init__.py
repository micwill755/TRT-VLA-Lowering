"""Registered VLA compile profiles."""

from __future__ import annotations

from trt.profile import VLAProfile

MODEL_REGISTRY: dict[str, str] = {
    "groot": "vla.profiles.groot:GrootProfile",
    "pi05": "vla.profiles.pi05:Pi05Profile",
    "smolvla": "vla.profiles.smolvla:SmolVLAProfile",
    "molmoact2": "vla.profiles.molmoact2:MolmoAct2Profile",
}


def get_profile_class(name: str) -> type[VLAProfile]:
    try:
        target = MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise SystemExit(f"Model not supported: {name!r}. Choose from: {known}") from exc

    module_path, class_name = target.split(":")
    from importlib import import_module

    module = import_module(module_path)
    return getattr(module, class_name)


def get_profile(name: str) -> type[VLAProfile]:
    """Return the profile class (instantiate with device + model_id in the orchestrator)."""
    return get_profile_class(name)
