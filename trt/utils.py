import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

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
def build_prefix_inputs(core, image_embs, img_masks, tokens, masks):
    # 1. Build one long prefix embedding sequence:
    #    image patch embeddings from the vision tower, followed by language token embeddings.
    embs = []
    pad_masks = []
    att_masks = []

    for img_emb, img_mask in zip(image_embs, img_masks, strict=True):
        bsz, n_img = img_emb.shape[:2]
        embs.append(img_emb)

        # 2. Build prefix_pad_masks:
        #    if an image is present, all of its patch tokens are valid.
        pad_masks.append(img_mask[:, None].expand(bsz, n_img))
        att_masks += [0] * n_img

    lang_emb = core.paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)
    pad_masks.append(masks)
    att_masks += [0] * lang_emb.shape[1]

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)

    # 3. Build the prefix attention mask:
    #    all-zero att_masks make all valid prefix tokens mutually visible.
    prefix_att_masks = torch.tensor(
        att_masks,
        dtype=torch.bool,
        device=prefix_embs.device,
    )[None, :].expand(prefix_embs.shape[0], -1)

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_attention_mask = core._prepare_attention_masks_4d(prefix_att_2d_masks)

    # 4. Build position ids:
    #    valid tokens count upward; padding does not advance the position.
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    return prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids

@torch.no_grad()
def compact_prefix_inputs(prefix_embs, prefix_pad_masks, position_ids):
    # 1. Find the real prefix tokens.
    #    prefix_pad_masks is True for image/language tokens that should remain.
    valid = prefix_pad_masks.to(device=prefix_embs.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)

    # 2. TensorRT needs a rectangular batch, so every batch item must compact
    #    down to the same number of valid prefix tokens.
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "compact_prefix_inputs requires equal valid token counts across the batch"
        )

    compact_len = int(valid_counts[0].item())

    # 3. Physically remove padded tokens from embeddings and position ids.
    #    After this, there are no padded prefix slots left to attend to.
    compact_embs = torch.stack(
        [prefix_embs[b, valid[b], :] for b in range(prefix_embs.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [position_ids[b, valid[b]] for b in range(position_ids.shape[0])],
        dim=0,
    )

    # 4. Rebuild masks for the compact sequence.
    #    Since every remaining token is valid and prefix attention is fully visible,
    #    the pad mask is all True and the 4D attention mask is all zeros.
    compact_pad_masks = torch.ones(
        prefix_embs.shape[0],
        compact_len,
        device=prefix_pad_masks.device,
        dtype=torch.bool,
    )
    compact_attention_mask = torch.zeros(
        prefix_embs.shape[0],
        1,
        compact_len,
        compact_len,
        device=prefix_embs.device,
        dtype=torch.float32,
    )
    return compact_embs, compact_pad_masks, compact_attention_mask, compact_position_ids

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