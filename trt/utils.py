import copy
import gc
from typing import Any

import torch
import torch.nn as nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

from trt.packing import compact_pi05_prefix

# hugging face utils ----

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
 
# hugging face utils ----

# utils ----

def free_cuda_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def prepare_policy_inputs_groot(policy, batch, device):
    allowed_base = {"state", "state_mask", "embodiment_id"}
    model_inputs = {
        k: v
        for k, v in batch.items()
        if (k in allowed_base or k.startswith("eagle_"))
        and not (k.startswith("next.") or k == "info")
    }

    return policy._groot_model.to(device).eval().prepare_input(model_inputs)

def prepare_policy_inputs(policy, batch, device):
    # returns four arguments that PI0.5’s core model needs for inference
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
    return images, img_masks, tokens, masks

@torch.no_grad()
def compact_prefix_inputs(prefix_embs, prefix_pad_masks, position_ids):
    packed = compact_pi05_prefix({
        "inputs_embeds": prefix_embs,
        "pad_mask": prefix_pad_masks,
        "position_ids": position_ids,
    })
    return (
        packed["inputs_embeds"],
        packed["pad_mask"],
        packed["attention_mask"],
        packed["position_ids"],
    )

@torch.no_grad()
def sample_actions_eager(policy, batch, noise, num_steps, device):
    core = policy.model
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)
    return core.sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        noise=noise.clone(),
        num_steps=num_steps,
    )
    
def make_suffix_position_and_mask(core, prefix_pad_masks, x_t, device):
    batch_size, suffix_len = x_t.shape[:2]
    prefix_pad_masks = prefix_pad_masks.to(device=device)
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + [0] * (suffix_len - 1),
        dtype=torch.int64,
        device=device,
    )[None, :].expand(batch_size, -1)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    attention_mask = core._prepare_attention_masks_4d(full_att_2d_masks)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask

def make_runner_inputs(core, prefix_pad_masks, prefix_k, prefix_v, x_t, timestep, device):
    action_dtype = next(core.action_in_proj.parameters()).dtype
    position_ids, attention_mask = make_suffix_position_and_mask(core, prefix_pad_masks, x_t, device)

    return (
        x_t.to(device=device, dtype=action_dtype),
        timestep.to(device=device),
        prefix_k.to(device=device, dtype=action_dtype),
        prefix_v.to(device=device, dtype=action_dtype),
        position_ids,
        attention_mask,
    )


_OPENPI_ATTENTION_MASK_NEG = -2.3819763e38


def bool_attention_mask_to_pi05_float(mask: torch.Tensor) -> torch.Tensor:
    """Convert bool [B, Q, K] masks to PI0.5 / Edge-LLM float additive masks."""
    return torch.where(
        mask,
        torch.zeros((), dtype=torch.float32, device=mask.device),
        torch.tensor(_OPENPI_ATTENTION_MASK_NEG, dtype=torch.float32, device=mask.device),
    )


def make_smolvla_suffix_position_and_mask(prefix_pad_masks, x_t, device, *, edge_llm: bool = False):
    """Suffix position/mask wiring matching SmolVLA ``denoise_step``."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    batch_size, suffix_len = x_t.shape[:2]
    prefix_pad_masks = prefix_pad_masks.to(device=device)
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    attention_mask = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    if edge_llm:
        attention_mask = bool_attention_mask_to_pi05_float(attention_mask)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask


def make_smolvla_runner_inputs(
    core,
    prefix_pad_masks,
    prefix_k,
    prefix_v,
    x_t,
    timestep,
    device,
    *,
    edge_llm: bool = False,
):
    action_dtype = next(core.action_in_proj.parameters()).dtype
    position_ids, attention_mask = make_smolvla_suffix_position_and_mask(
        prefix_pad_masks, x_t, device, edge_llm=edge_llm
    )

    return (
        x_t.to(device=device, dtype=action_dtype),
        timestep.to(device=device),
        prefix_k.to(device=device, dtype=action_dtype),
        prefix_v.to(device=device, dtype=action_dtype),
        position_ids,
        attention_mask,
    )

# pi05 utils ----

def clone_hf_module_for_export(
    module: nn.Module,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    config: Any | None = None,
) -> nn.Module:
    """Build a disposable copy of an HF submodule for TRT export only.

    **Do not use the live policy module for export.** One ``EdgeContext`` keeps
    ``ctx.policy`` loaded for the whole orchestrator run: later export stages
    (language, action), eager PyTorch benchmarking, and parity checks all still
    need the original weights on GPU and unmodified.

    Export is destructive on the trace target:

    - ``compile()`` hooks patch attention layers (``patch_vision_attention`` /
      ``patch_language_attention``) so torch→TRT can capture custom ops.
    - TRT settings use ``offload_module_to_cpu=True``, moving the traced module
      to CPU during engine build to free GPU for TensorRT.
    - ``ExportRunner`` deletes ``ExportPlan.cleanup_modules`` after the engine
      is written (``free_cuda_memory``).

    Clones are listed in ``cleanup_modules`` so export can patch, offload, and
    free GPU memory without touching ``ctx.policy``. Inference loads serialized
    engines only — it never uses clones.

    **Memory:** weights are copied to CPU first, CUDA cache is cleared, then a
    fresh module is instantiated from ``config`` (or ``deepcopy`` fallback) and
    loaded back on ``device`` / ``dtype``. This avoids a peak GPU ``deepcopy`` of
    large vision / LM weights while duplicating the subgraph for trace.
    """
    # Snapshot weights on CPU and release GPU before rebuilding the clone.
    cpu_state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    free_cuda_memory()

    init_config = config if config is not None else getattr(module, "config", None)
    if init_config is not None:
        # Preferred path for HF modules: new instance from config + state_dict.
        clone = module.__class__(init_config)
        clone.load_state_dict(cpu_state, assign=True)
    else:
        # Fallback when the module has no ``config`` (rare submodules).
        clone = copy.deepcopy(module)
        clone.load_state_dict(cpu_state, assign=True)
        clone = clone.cpu()
    del cpu_state

    if dtype is not None:
        return clone.to(device=device, dtype=dtype).eval()
    return clone.to(device=device).eval()


def ensure_pi05_paligemma_on_device(core: nn.Module, device: torch.device) -> None:
    """Move PaliGemma weights back to GPU after TRT export/offload touched shared modules."""
    paligemma = core.paligemma_with_expert.paligemma.model
    paligemma.vision_tower.to(device=device)
    paligemma.multi_modal_projector.to(device=device)
    paligemma.language_model.to(device=device)


def ensure_smolvla_on_device(core: nn.Module, device: torch.device) -> None:
    """Move SmolVLA weights back to GPU after TRT export/offload touched shared modules."""
    core.vlm_with_expert.to(device=device)
    core.state_proj.to(device=device)
    core.action_in_proj.to(device=device)
    core.action_time_mlp_in.to(device=device)
    core.action_time_mlp_out.to(device=device)
    core.action_out_proj.to(device=device)