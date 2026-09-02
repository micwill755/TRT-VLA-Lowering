"""PI05 through EdgeExporter: hybrid VLM graph + action expert.

Same load / libero frame / preprocessor path as ``vla/test_vla_pi05_e2e.py``.

    exporter.export_for_policy(...)   # vision | fuse | language
    exporter.export(action_module, ...)  # action_expert
    exporter.save_engines(out_dir)    # serialized engines + config.json

Run::

    python graph/examples/pi05.py
    python graph/examples/pi05.py --compile
    python graph/examples/pi05.py --compile --save /tmp/pi05_engines
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
_GRAPH_DIR = _EXAMPLES_DIR.parent
for _path in (_GRAPH_DIR, _EXAMPLES_DIR):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import _common as common

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from exporter import EdgeExporter, dump_graph
from models.pi05 import wrap_action
from models.pi05.patches.vision import wrap_vision
from trt.data import frame_from_test_data, load_test_data
from trt.executor.models.pi05.helpers import make_pi05_suffix_position_and_mask
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.rope import make_rope_rotary_cos_sin
from trt.utils import configure_thor_pytorch, force_hf_attention

configure_thor_pytorch()


def load_config():
    policy = PI05Policy.from_pretrained("lerobot/pi05_libero_base").eval()
    config = policy.config
    config.device = "cpu"
    config.chunk_size = 50
    config.n_action_steps = 50
    config.max_state_dim = 32
    config.max_action_dim = 32
    config.input_features = {
        f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.image3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.image4": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))}
    config.empty_cameras = 0
    config.validate_features()
    return config, policy


def main() -> None:
    args = common.parse_args(__doc__)
    device = torch.device("cuda")
    dtype = torch.float16
    common.ensure_plugin_so()
    load_plugins_for_trt()

    config, policy = load_config()
    model = policy.model.to(device=device, dtype=dtype).eval()
    paligemma = model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    language = paligemma.language_model
    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")
    force_hf_attention(model.paligemma_with_expert.gemma_expert.model, "eager")

    pre_processor, _ = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    data = load_test_data("lerobot/libero", episode_index=0, frame_index=0)
    frame = frame_from_test_data(data, policy, fill_missing=True)
    model_inputs = pre_processor(frame)
    images, _img_masks = policy._preprocess_images(model_inputs)
    tokens = model_inputs[OBS_LANGUAGE_TOKENS].to(device=device, dtype=torch.long)

    pixel_values = torch.cat(
        [img.to(device=device, dtype=dtype) for img in images],
        dim=0,
    ).contiguous()
    lang_embeds = model.paligemma_with_expert.embed_language_tokens(tokens)
    lang_embeds = lang_embeds.to(device=device, dtype=dtype).contiguous()
    batch = int(lang_embeds.shape[0])

    vision_mod = wrap_vision(model, {"pixel_values": pixel_values})
    vision_seq = int(vision_mod.output_num_tokens) // batch
    fused_seq = vision_seq + int(lang_embeds.shape[1])
    print(f"vision tokens={vision_seq}  lang tokens={lang_embeds.shape[1]}  fused seq={fused_seq}")

    decoder = getattr(language, "model", language)
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(decoder.layers)

    rope = make_rope_rotary_cos_sin(cfg, fused_seq, device, language_model=language)
    ctx_len = torch.full((batch,), fused_seq, device=device, dtype=torch.int32)
    last_token_ids = torch.full((batch, 1), fused_seq - 1, device=device, dtype=torch.int64)
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    ds_stack = torch.zeros(0, batch, fused_seq, hidden_size, device=device, dtype=dtype)
    past_key_values = [
        torch.zeros(
            batch, 2, num_key_value_heads, fused_seq, head_dim, device=device, dtype=dtype
        )
        for _ in range(num_layers)
    ]

    inputs = {
        "pixel_values": pixel_values,
        "lang_embeds": lang_embeds,
        "rope_rotary_cos_sin": rope,
        "context_lengths": ctx_len,
        "kvcache_start_index": kvcache_start_index,
        "last_token_ids": last_token_ids,
        "ds_stack": ds_stack,
        "past_key_values": past_key_values,
    }

    exporter = EdgeExporter()
    compiled = exporter.export_for_policy(
        model, inputs, config=common.edge_config(compile_engines=args.compile)
    )
    common.dump_partition(compiled, "PI05 PolicyStep partition")
    if args.compile:
        dump_graph(compiled, "graph 3  edgellm.vision_tower | fuse | text_encoder")

    prefix_k = torch.zeros(
        num_layers, batch, num_key_value_heads, fused_seq, head_dim, device=device, dtype=dtype
    )
    prefix_v = torch.zeros_like(prefix_k)
    step_actions = torch.randn(
        batch, int(model.config.chunk_size), int(model.config.max_action_dim),
        device=device, dtype=dtype,
    )
    step_timestep = torch.full((batch,), 1.0, device=device, dtype=torch.float32)
    prefix_pad = torch.ones(batch, fused_seq, dtype=torch.bool, device=device)
    suffix_position_ids, suffix_attention_mask = make_pi05_suffix_position_and_mask(
        model, prefix_pad, step_actions, device
    )
    action_inputs = (
        step_actions,
        step_timestep,
        prefix_k,
        prefix_v,
        suffix_position_ids,
        suffix_attention_mask,
    )
    action_module = wrap_action(model).to(device=device)
    expert = exporter.export(
        action_module,
        action_inputs,
        config=common.edge_config(compile_engines=args.compile, full=True),
        components=("action",),
    )
    common.dump_partition(expert, "PI05 action_expert partition")
    if args.compile:
        dump_graph(expert, "graph 3  edgellm.action_expert")

    if args.save:
        written = exporter.save_engines(args.save)
        for name, path in written.items():
            print(f"saved {name}: {path}")


if __name__ == "__main__":
    main()
