import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from trt.packing import (
    MultimodalPromptProcessor,
    PackedLanguageInputs,
    PromptPackingSpec,
    PromptTensorInputs,
    compact_packed_language_inputs,
)

def disable_flash_attention(module):
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "eager"
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = "eager"

    cfg = getattr(module, "config", None)
    if cfg is not None:
        for name in ("vision_config", "text_config"):
            sub_cfg = getattr(cfg, name, None)
            if sub_cfg is not None:
                if hasattr(sub_cfg, "_attn_implementation"):
                    sub_cfg._attn_implementation = "eager"
                if hasattr(sub_cfg, "attn_implementation"):
                    sub_cfg.attn_implementation = "eager"

# hugging face utils ----

def load_policy(policy_cls, model_id, device, disable_flash_attn=True):
    policy = policy_cls.from_pretrained(model_id, device=device)
    if disable_flash_attn:
        disable_flash_attention(policy)
    return policy

# hugging face utils ----

# utils ----

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
def build_packed_prefix_inputs(core, image_embs, img_masks, tokens, masks) -> PackedLanguageInputs:
    processor = MultimodalPromptProcessor(
        PromptPackingSpec(
            style="concat_prefix",
            make_att_2d_masks=make_att_2d_masks,
            prepare_attention_mask_4d=core._prepare_attention_masks_4d,
        )
    )

    return processor(
        PromptTensorInputs(
            image_embs=image_embs,
            image_masks=img_masks,
            text_embs=core.paligemma_with_expert.embed_language_tokens(tokens),
            text_mask=masks,
        )
    )

@torch.no_grad()
def build_prefix_inputs(core, image_embs, img_masks, tokens, masks):
    return build_packed_prefix_inputs(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
    ).as_tuple()

@torch.no_grad()
def compact_prefix_inputs(prefix_embs, prefix_pad_masks, position_ids):
    packed = PackedLanguageInputs(
        inputs_embeds=prefix_embs,
        pad_mask=prefix_pad_masks,
        position_ids=position_ids,
    )
    return compact_packed_language_inputs(packed).as_tuple()

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

# pi05 utils ----