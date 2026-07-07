import gc
from typing import Any

import torch


def force_hf_attention(module, attn):
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = attn
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = attn

    cfg = getattr(module, "config", None)
    if cfg is not None:
        for name in ("vision_config", "text_config"):
            sub_cfg = getattr(cfg, name, None)
            if sub_cfg is not None:
                if hasattr(sub_cfg, "_attn_implementation"):
                    sub_cfg._attn_implementation = attn
                if hasattr(sub_cfg, "attn_implementation"):
                    sub_cfg.attn_implementation = attn


def find_pack_step(preprocessor):
    for step in preprocessor.steps:
        if step.__class__.__name__ == "MolmoAct2PackInputsProcessorStep":
            return step
    raise ValueError("MolmoAct2PackInputsProcessorStep not found in preprocessor pipeline")


def free_cuda_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
