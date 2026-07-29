"""Checkpoint split and assignment for the standalone-model approach."""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from safetensors.torch import load_file

_UND_ATTN_RENAME = {
    ".self_attn.to_q.": ".self_attn.q_proj.",
    ".self_attn.to_k.": ".self_attn.k_proj.",
    ".self_attn.to_v.": ".self_attn.v_proj.",
    ".self_attn.to_out.": ".self_attn.o_proj.",
    ".self_attn.norm_q.": ".self_attn.q_norm.",
    ".self_attn.norm_k.": ".self_attn.k_norm.",
}

_GEN_LAYER_MARKERS = (
    ".self_attn.add_q_proj.",
    ".self_attn.add_k_proj.",
    ".self_attn.add_v_proj.",
    ".self_attn.to_add_out.",
    ".self_attn.norm_added_q.",
    ".self_attn.norm_added_k.",
    ".mlp_moe_gen.",
    ".input_layernorm_moe_gen.",
    ".post_attention_layernorm_moe_gen.",
)

_GEN_TOPLEVEL = (
    "proj_in.",
    "proj_out.",
    "time_embedder.",
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
    "norm_moe_gen.",
)

_GEN_LAYER_RENAME = {
    ".self_attn.add_q_proj.": ".cross_attention.to_q.",
    ".self_attn.add_k_proj.": ".cross_attention.to_k.",
    ".self_attn.add_v_proj.": ".cross_attention.to_v.",
    ".self_attn.to_add_out.": ".cross_attention.to_out.",
    ".self_attn.norm_added_q.": ".cross_attention.norm_q.",
    ".self_attn.norm_added_k.": ".cross_attention.norm_k.",
    ".mlp_moe_gen.": ".mlp.",
    ".input_layernorm_moe_gen.": ".input_layernorm.",
    ".post_attention_layernorm_moe_gen.": ".post_attention_layernorm.",
}


def _rename(key: str, table: dict[str, str]) -> str:
    for src, dst in table.items():
        if src in key:
            return key.replace(src, dst)
    return key


def split_transformer_weights(
    transformer_dir: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(str(Path(transformer_dir) / "*.safetensors"))):
        raw.update(load_file(shard))
    if not raw:
        raise FileNotFoundError(f"No safetensors in {transformer_dir}")

    und: dict[str, torch.Tensor] = {}
    gen: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if any(key.startswith(prefix) for prefix in _GEN_TOPLEVEL):
            if key.startswith(("action_proj_in.", "action_proj_out.")):
                key = key.replace(".fc.weight", ".fc").replace(".bias.weight", ".bias")
            gen[key] = tensor
        elif any(marker in key for marker in _GEN_LAYER_MARKERS):
            gen[_rename(key, _GEN_LAYER_RENAME)] = tensor
        else:
            und[_rename(key, _UND_ATTN_RENAME)] = tensor
    return und, gen


def load_und_weights(module, weights: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    state = {
        key: tensor.to(dtype) if tensor.is_floating_point() else tensor
        for key, tensor in weights.items()
        if not key.startswith(("embed_tokens.", "lm_head."))
    }
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise KeyError(
            f"UND load mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )


def load_gen_weights(module, weights: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    cfg = module.cfg
    domain = cfg.domain_id
    assigned: set[str] = set()

    def assign(path: str, tensor: torch.Tensor) -> None:
        obj = module
        parts = path.split(".")
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        param = getattr(obj, parts[-1])
        param.data.copy_(tensor.to(dtype) if tensor.is_floating_point() else tensor)
        assigned.add(path)

    for key, tensor in weights.items():
        if key == "action_modality_embed":
            assign("action_modality_embed", tensor.reshape(-1))
        elif key in ("action_proj_in.fc", "action_proj_in.fc.weight"):
            assign(
                "action_in_weight",
                tensor.reshape(cfg.num_embodiment_domains, cfg.max_action_dim, cfg.hidden_size)[domain],
            )
        elif key in ("action_proj_in.bias", "action_proj_in.bias.weight"):
            assign(
                "action_in_bias",
                tensor.reshape(cfg.num_embodiment_domains, cfg.hidden_size)[domain],
            )
        elif key in ("action_proj_out.fc", "action_proj_out.fc.weight"):
            assign(
                "action_out_weight",
                tensor.reshape(cfg.num_embodiment_domains, cfg.hidden_size, cfg.max_action_dim)[domain],
            )
        elif key in ("action_proj_out.bias", "action_proj_out.bias.weight"):
            assign(
                "action_out_bias",
                tensor.reshape(cfg.num_embodiment_domains, cfg.max_action_dim)[domain],
            )
        else:
            assign(key, tensor)

    missing = sorted(name for name, _ in module.named_parameters() if name not in assigned)
    if missing:
        raise KeyError(f"GEN parameters received no checkpoint tensor: {missing[:8]}")
