from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from . import root


def discover(policy: nn.Module) -> nn.Module:
    found = root(policy)
    if found is None:
        raise RuntimeError("PI05 language adapter matched a policy without paligemma_with_expert")
    return found.paligemma_with_expert.paligemma.model.language_model.eval()


def wrap_language(policy: nn.Module, _inputs: Mapping[str, Any]) -> nn.Module:
    from trt.modules.export.language import CausalLMExportModule

    found = root(policy)
    if found is None:
        raise RuntimeError("PI05 language patch matched a policy without paligemma_with_expert")
    paligemma_root = found.paligemma_with_expert.paligemma
    language = paligemma_root.model.language_model
    decoder = getattr(language, "model", language)
    lm_head = getattr(paligemma_root, "lm_head", None) or getattr(policy, "lm_head", None)
    if lm_head is None:
        raise AttributeError("policy has no lm_head; cannot adapt language for Edge export")
    return CausalLMExportModule(decoder, lm_head).eval()
