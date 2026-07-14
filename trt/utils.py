import gc
import os
from pathlib import Path
from typing import Any

import torch

_THOR_CUDA_LIB = Path("/usr/local/cuda-13.0/thor/targets/aarch64-linux/lib")


def configure_thor_pytorch() -> None:
    """Use PyTorch fallbacks for ops whose pip CUDA wheels mismatch DriveOS Thor."""
    on_thor = os.environ.get("TRT_VLA_THOR", "auto")
    if on_thor == "auto":
        on_thor = "1" if _THOR_CUDA_LIB.is_dir() else "0"
    if on_thor == "1":
        torch.backends.cudnn.enabled = False


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


def release_serialized_trt_engine(serialized: Any) -> None:
    """Drop TensorRT runtime/context GPU allocations before the next compile stage."""
    inner = getattr(serialized, "engine", serialized)
    for attr in ("context", "engine", "runtime"):
        if hasattr(inner, attr):
            delattr(inner, attr)
    if hasattr(inner, "_zero_size_input_dummy"):
        inner._zero_size_input_dummy.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def move_pi05_diffusion_modules_to_device(
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model.paligemma_with_expert.gemma_expert.to(device=device, dtype=dtype)
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        getattr(model, name).to(device=device, dtype=dtype)
