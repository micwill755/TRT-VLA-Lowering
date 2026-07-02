from __future__ import annotations

from trt.hooks.resolve import resolve
from trt.profile.base import VLAProfile

_PROFILE_PATHS: dict[str, str] = {
    "gr00t": "trt.profile.groot:GrootProfile",
    "pi05": "trt.profile.pi05:Pi05Profile",
    "smolvla": "trt.profile.smolvla:SmolVLAProfile",
    "molmo2": "trt.profile.molmoact2:MolmoAct2Profile",
}

MODEL_REGISTRY: dict[str, str] = dict(_PROFILE_PATHS)

def get_profile(name: str) -> type[VLAProfile]:
    key = name.strip().lower()
    path = _PROFILE_PATHS.get(key)
    if path is None:
        known = ", ".join(sorted(_PROFILE_PATHS))
        raise KeyError(f"Unknown profile {name!r}. Known: {known}")
    return resolve(path)
