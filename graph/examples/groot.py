"""GR00T through EdgeExporter: hybrid VLM graph + action context + action expert.

Same load / Eagle chat template / libero frame path as ``vla/test_vla_gr00t_e2e.py``.

    exporter.export_for_policy(...)   # vision | scatter | language
    exporter.export(action_context, ...)  # action_context
    exporter.export(action_module, ...)   # action_expert
    exporter.save_engines(out_dir)

Run::

    python graph/examples/groot.py
    python graph/examples/groot.py --compile
    python graph/examples/groot.py --compile --save /tmp/groot_engines
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
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep
from lerobot.utils.constants import ACTION, OBS_STATE

from exporter import EdgeExporter, dump_graph
from models.groot import wrap_action, wrap_action_context
from trt.data import create_pil_messages, load_test_data
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.rope import make_rope_rotary_cos_sin
from trt.utils import configure_thor_pytorch, force_hf_attention

configure_thor_pytorch()


def load_config(device: torch.device):
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.5-3B",
        device=str(device),
        embodiment_tag="new_embodiment",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=64,
        max_action_dim=32,
        image_size=(224, 224),
        tokenizer_assets_repo="lerobot/eagle2hg-processor-groot-n1p5",
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    )
    policy = GrootPolicy(config).to(device).eval()
    return config, policy


def main() -> None:
    args = common.parse_args(__doc__)
    device = torch.device("cuda")
    dtype = torch.float16
    common.ensure_plugin_so()
    load_plugins_for_trt()

    config, policy = load_config(device)
    model = policy._groot_model
    eagle = model.backbone.eagle_model
    vision = eagle.vision_model
    language = eagle.language_model
    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")
    language.config._attn_implementation = "sdpa"
    language = language.to(device=device, dtype=dtype).eval()

    pre_processor, _ = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    eagle_step = next(s for s in pre_processor.steps if isinstance(s, GrootEagleEncodeStep))
    eagle_processor = eagle_step.proc

    data = load_test_data("lerobot/libero", episode_index=0, frame_index=0)
    messages = create_pil_messages(data)
    text = eagle_processor.apply_chat_template(
        messages,
        tokenize=False,
        **{"add_generation_prompt": True},
    )
    image_inputs, video_inputs = eagle_processor.process_vision_info(messages)
    tokenized_data = eagle_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **{
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            }
        },
    )

    input_ids = tokenized_data["input_ids"].to(device=device, dtype=torch.long)
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=dtype)

    vocab_size = int(language.get_input_embeddings().num_embeddings)
    safe_ids = torch.where(input_ids >= vocab_size, torch.zeros_like(input_ids), input_ids)
    lang_embeds = language.get_input_embeddings()(safe_ids).to(dtype=dtype).contiguous()
    image_token_index = getattr(eagle, "image_token_index", eagle.config.image_token_index)
    image_token_mask = (input_ids == image_token_index) | (input_ids >= vocab_size)

    bsz, seq_len, _hidden = lang_embeds.shape
    n_image_slots = int(image_token_mask.sum().item())
    print(f"prompt seq={seq_len}  image slots={n_image_slots}  (scatter, not cat)")

    decoder = getattr(language, "model", language)
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(decoder.layers)

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    rope = make_rope_rotary_cos_sin(
        cfg, seq_len, device, language_model=language, position_ids=position_ids
    )
    ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.int64)
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    ds_stack = torch.zeros(0, bsz, seq_len, hidden_size, device=device, dtype=dtype)
    past_key_values = [
        torch.zeros(
            bsz, 2, num_key_value_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        for _ in range(num_layers)
    ]

    inputs = {
        "pixel_values": pixel_values,
        "lang_embeds": lang_embeds,
        "image_token_mask": image_token_mask,
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
    common.dump_partition(compiled, "GR00T PolicyStep partition")
    if args.compile:
        dump_graph(compiled, "graph 3  edgellm.vision_tower | scatter | text_encoder")

    action_head = model.action_head
    action_horizon = int(action_head.config.action_horizon)
    action_dim = int(action_head.config.action_dim)
    # Dummy language hidden — action_context input is [B, S, H_lm], not projected.
    lm_hidden = torch.zeros(bsz, seq_len, hidden_size, device=device, dtype=dtype)
    context_module = wrap_action_context(model).to(device=device, dtype=dtype)
    context = exporter.export(
        context_module,
        (lm_hidden,),
        config=common.edge_config(compile_engines=args.compile, full=True),
        components=("action_context",),
    )
    common.dump_partition(context, "GR00T action_context partition")
    if args.compile:
        dump_graph(context, "graph 3  edgellm.action_context")

    with torch.no_grad():
        context_embs = torch.zeros_like(context_module(lm_hidden))
    step_actions = torch.randn(bsz, action_horizon, action_dim, device=device, dtype=dtype)
    step_timestep = torch.zeros(bsz, device=device, dtype=torch.long)
    state = torch.zeros(bsz, 1, int(config.max_state_dim), device=device, dtype=dtype)
    embodiment_id = torch.zeros(bsz, device=device, dtype=torch.long)
    action_inputs = (step_actions, step_timestep, context_embs, state, embodiment_id)
    action_module = wrap_action(
        model, {"embodiment_id": embodiment_id}
    ).to(device=device, dtype=dtype)
    expert = exporter.export(
        action_module,
        action_inputs,
        config=common.edge_config(compile_engines=args.compile, full=True),
        components=("action",),
    )
    common.dump_partition(expert, "GR00T action_expert partition")
    if args.compile:
        dump_graph(expert, "graph 3  edgellm.action_expert")

    if args.save:
        written = exporter.save_engines(args.save)
        for name, path in written.items():
            print(f"saved {name}: {path}")


if __name__ == "__main__":
    main()
