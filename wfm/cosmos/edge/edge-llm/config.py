from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Cosmos3GenConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    rms_norm_eps: float = 1e-6
    hidden_act: str = "relu2"
    latent_channel: int = 48
    latent_patch_size: int = 2
    max_action_dim: int = 64
    action_chunk_size: int = 16
    num_embodiment_domains: int = 32
    frequency_embedding_size: int = 256
    timestep_max_period: int = 10000
    timestep_scale: float = 0.001
    latent_t: int = 5
    latent_h: int = 34
    latent_w: int = 46
    domain_id: int = 8

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size * self.latent_patch_size * self.latent_channel

    @property
    def hp(self) -> int:
        return (self.latent_h + self.latent_patch_size - 1) // self.latent_patch_size

    @property
    def wp(self) -> int:
        return (self.latent_w + self.latent_patch_size - 1) // self.latent_patch_size

    @property
    def num_video_tokens(self) -> int:
        return self.latent_t * self.hp * self.wp

    @property
    def num_gen_tokens(self) -> int:
        return self.num_video_tokens + self.action_chunk_size


ACTION_CHUNK_SIZE = 16
DEFAULT_NUM_FRAMES = 17
DEFAULT_FPS = 5.0
# droid_lerobot raw action head: 3 position + 6D rotation + gripper = 10 dims
# (the 8-value DROID control action is derived downstream by converting the 6D
# rotation to Euler angles). Dims >= raw_action_dim are zeroed padding.
DEFAULT_RAW_ACTION_DIM = 10
DEFAULT_DOMAIN = "droid_lerobot"
DEFAULT_DOMAIN_ID = 8
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_FLOW_SHIFT = 5.0


def und_config_from_transformer(tcfg: dict) -> dict:
    return {
        "hidden_size": tcfg["hidden_size"],
        "num_hidden_layers": tcfg["num_hidden_layers"],
        "num_attention_heads": tcfg["num_attention_heads"],
        "num_key_value_heads": tcfg["num_key_value_heads"],
        "head_dim": tcfg["head_dim"],
        "intermediate_size": tcfg["intermediate_size"],
        "hidden_act": str(tcfg["hidden_act"]),
        "rope_theta": float(tcfg["rope_theta"]),
        "rms_norm_eps": tcfg.get("rms_norm_eps", 1e-6),
        "use_und_k_norm_for_gen": bool(tcfg.get("use_und_k_norm_for_gen", False)),
    }


def make_vae_encoder_config(height: int, width: int, num_frames: int) -> dict:
    shape = [1, 3, num_frames, height, width]
    return {
        "component": "vae_encoder",
        "onnx_filename": "model.onnx",
        "engine_filename": "vae_encoder.engine",
        "optimization_profile": {
            "pixel_values": {"min": shape, "opt": shape, "max": shape},
        },
        "tensor_contract": {
            "inputs": {"pixel_values": ["batch", 3, num_frames, height, width]},
            "outputs": {"cond_latent": ["batch", "latent_channel", "t", "h", "w"]},
        },
        "builder_config": {
            "max_batch_size": 1,
            "height": height,
            "width": width,
            "num_frames": num_frames,
        },
    }


def make_und_prefill_config(cfg: dict, max_und_len: int) -> dict:
    hd = cfg["head_dim"]
    hs = cfg["hidden_size"]
    return {
        "component": "und_prefill",
        "onnx_filename": "model.onnx",
        "engine_filename": "und_prefill.engine",
        "optimization_profile": {
            "inputs_embeds": {
                "min": [1, 2, hs],
                "opt": [1, 16, hs],
                "max": [1, max_und_len, hs],
            },
            "rope_rotary_cos_sin": {
                "min": [1, 2, hd],
                "opt": [1, 16, hd],
                "max": [1, max_und_len, hd],
            },
            "attention_pos_id": {
                "min": [1, 2],
                "opt": [1, 16],
                "max": [1, max_und_len],
            },
        },
        "tensor_contract": {
            "inputs": {
                "inputs_embeds": ["batch", "und_len", "hidden_size"],
                "rope_rotary_cos_sin": ["batch", "und_len", "head_dim"],
                "attention_pos_id": ["batch", "und_len"],
            },
            "outputs": {
                "und_k_layerNN": ["batch", "und_len", "num_kv_heads", "head_dim"],
                "und_v_layerNN": ["batch", "und_len", "num_kv_heads", "head_dim"],
                "hidden_states": ["batch", "und_len", "hidden_size"],
            },
        },
        "builder_config": {"max_batch_size": 1, "max_und_len": max_und_len},
        "num_hidden_layers": cfg["num_hidden_layers"],
        "hidden_size": hs,
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": hd,
        "rope_theta": cfg["rope_theta"],
    }

def gen_config_from_transformer(
        tcfg: dict,
        action_chunk_size: "int | None" = None,
        num_frames: "int | None" = None) -> Cosmos3GenConfig:
    """Build the GEN config object from a Cosmos3 transformer config.

    ``action_chunk_size`` / ``num_frames`` are request-time parameters (the
    canonical policy request uses 16 / 17); ``num_frames`` sets the temporal
    latent extent ``latent_t = (num_frames - 1) // 4 + 1``.
    """
    if num_frames is not None and (num_frames - 1) % 4 != 0:
        raise ValueError(
            f"num_frames must be 4k+1 (got {num_frames}); the VAE compresses "
            "time 4x with a single leading conditioning frame.")
    return Cosmos3GenConfig(
        hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        rms_norm_eps=tcfg.get("rms_norm_eps", 1e-6),
        # Required: the activation defines the MLP graph (Cosmos3-Edge uses
        # the Nemotron-H squared-ReLU 'relu2'); no silent default.
        hidden_act=str(tcfg["hidden_act"]),
        latent_channel=tcfg.get("latent_channel", 48),
        latent_patch_size=tcfg.get("latent_patch_size", 2),
        max_action_dim=tcfg.get("max_action_dim", 64),
        num_embodiment_domains=tcfg.get("num_embodiment_domains", 32),
        timestep_scale=tcfg.get("timestep_scale", 0.001),
        action_chunk_size=(action_chunk_size
                           if action_chunk_size is not None else tcfg.get(
                               "action_chunk_size", ACTION_CHUNK_SIZE)),
        latent_t=((num_frames - 1) // 4 +
                  1 if num_frames is not None else Cosmos3GenConfig.latent_t),
        domain_id=tcfg.get("domain_id", DEFAULT_DOMAIN_ID),
    )


def make_gen_config(cfg: Any,
                    tcfg: dict,
                    max_und_len: int,
                    fps: float = DEFAULT_FPS) -> dict:
    """Return the GEN component ``config.json`` payload."""
    rope_scaling = tcfg.get("rope_scaling", {}) or {}
    v_tok = cfg.num_video_tokens
    g_tok = cfg.num_gen_tokens
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    action_len = cfg.action_chunk_size
    opt_und = min(32, max_und_len)

    def _fix(shape: list) -> dict:
        return {"min": shape, "opt": shape, "max": shape}

    profile = {
        "video_latent":
        _fix([1, cfg.latent_channel, cfg.latent_t, cfg.latent_h,
              cfg.latent_w]),
        "action_latent":
        _fix([1, action_len, cfg.max_action_dim]),
        "timestep":
        _fix([1]),
        "token_noisy_mask":
        _fix([1, v_tok, 1]),
        "action_noisy_mask":
        _fix([1, action_len, 1]),
        "rope_rotary_cos_sin":
        _fix([1, g_tok, hd]),
        "attention_pos_id":
        _fix([1, g_tok]),
    }
    for i in range(cfg.num_hidden_layers):
        und = {
            "min": [1, 1, n_kv, hd],
            "opt": [1, opt_und, n_kv, hd],
            "max": [1, max_und_len, n_kv, hd],
        }
        profile[f"und_k_layer{i:02d}"] = und
        profile[f"und_v_layer{i:02d}"] = und

    return {
        "component": "gen",
        "onnx_filename": "model.onnx",
        "engine_filename": "gen.engine",
        "optimization_profile": profile,
        "tensor_contract": {
            "inputs": {
                "video_latent": ["batch", "latent_channel", "t", "h", "w"],
                "action_latent":
                ["batch", "action_chunk_size", "max_action_dim"],
                "und_k_layerNN":
                ["batch", "und_len", "num_kv_heads", "head_dim"],
                "und_v_layerNN":
                ["batch", "und_len", "num_kv_heads", "head_dim"],
            },
            "outputs": {
                "video_pred": ["batch", "latent_channel", "t", "h", "w"],
                "action_pred":
                ["batch", "action_chunk_size", "max_action_dim"],
            },
        },
        "builder_config": {
            "max_batch_size": 1,
            "max_und_len": max_und_len,
            "num_und_kv_inputs": cfg.num_hidden_layers * 2,
        },
        # Required: rope geometry is architecture-defining; no silent defaults.
        "rope_theta": float(tcfg["rope_theta"]),
        "rope_scaling": {
            "mrope_section": rope_scaling["mrope_section"],
            "mrope_interleaved": rope_scaling.get("mrope_interleaved", True),
            "rope_type": "mrope",
        },
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "hidden_act": cfg.hidden_act,
        "latent_channel": cfg.latent_channel,
        "latent_patch_size": cfg.latent_patch_size,
        "num_video_tokens": cfg.num_video_tokens,
        "action_chunk_size": cfg.action_chunk_size,
        "raw_action_dim": DEFAULT_RAW_ACTION_DIM,
        "max_action_dim": cfg.max_action_dim,
        "num_embodiment_domains": cfg.num_embodiment_domains,
        "domain": DEFAULT_DOMAIN,
        "domain_id": cfg.domain_id,
        "timestep_scale": cfg.timestep_scale,
        "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
        "flow_shift": DEFAULT_FLOW_SHIFT,
        "video_latent_frames": cfg.latent_t,
        "fps": float(fps),
        "base_fps": 24.0,
        "temporal_compression_factor": 4,
        "temporal_modality_margin": 15000,
        "action_start_frame_offset": 1,
    }
