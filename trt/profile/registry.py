from __future__ import annotations

from trt.hooks.resolve import resolve
from trt.profile.base import VLAProfile

PROFILE_REGISTRY: dict[str, str] = {
    "gr00t": "trt.profile.groot:GrootProfile",
    "pi05": "trt.profile.pi05:Pi05Profile",
    "smolvla": "trt.profile.smolvla:SmolVLAProfile",
    "molmo2": "trt.profile.molmoact2:MolmoAct2Profile",
}

def get_profile(name: str) -> type[VLAProfile]:
    key = name.strip().lower()
    path = PROFILE_REGISTRY.get(key)
    if path is None:
        known = ", ".join(sorted(PROFILE_REGISTRY))
        raise KeyError(f"Unknown profile {name!r}. Known: {known}")
    return resolve(path)