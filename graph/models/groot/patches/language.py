from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from . import root


def discover(policy: nn.Module) -> nn.Module:
    found = root(policy)
    if found is None:
        raise RuntimeError("GR00T language adapter matched a policy without backbone.eagle_model")
    return found.backbone.eagle_model.language_model.eval()


def wrap_language(policy: nn.Module, _inputs: Mapping[str, Any]) -> nn.Module:
    from trt.modules.export.language import CausalLMExportModule

    found = root(policy)
    if found is None:
        raise RuntimeError("GR00T language patch matched a policy without backbone.eagle_model")
    language = found.backbone.eagle_model.language_model
    decoder = getattr(language, "model", language)
    return CausalLMExportModule(decoder, language.lm_head, select_layer=-1).eval()
