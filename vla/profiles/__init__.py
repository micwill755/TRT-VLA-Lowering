"""Registered VLA compile profiles."""

from __future__ import annotations

from vla.profile import VLAProfile
from vla.profiles.groot import GrootProfile
from vla.profiles.molmoact2 import MolmoAct2Profile
from vla.profiles.pi05 import Pi05Profile
from vla.profiles.smolvla import SmolVLAProfile

MODEL_REGISTRY: dict[str, type[VLAProfile]] = {
    GrootProfile.name: GrootProfile,
    Pi05Profile.name: Pi05Profile,
    SmolVLAProfile.name: SmolVLAProfile,
    MolmoAct2Profile.name: MolmoAct2Profile,
}

def get_profile(name: str) -> VLAProfile:
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise SystemExit(f"Unknown model {name!r}. Choose from: {known}") from exc
    return cls()
