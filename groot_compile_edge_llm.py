import torch
import copy
import json
import pathlib

from typing import Any

import torch_tensorrt

from transformers import AutoProcessor

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.groot_n1 import DEFAULT_TOKENIZER_ASSETS_REPO
from lerobot.utils.constants import HF_LEROBOT_HOME

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.compile import compile_trt_module
from trt.diffusion import GrootStaticDiffusionStep
from trt.utils import (
    load_policy,
    compact_prefix_inputs,
    prepare_policy_inputs_groot,
)
from trt.helper import (
    get_processor
)
from trt.data import (
    load_test_data,
    prepare_model_inputs,
    make_batch,
    pack_state
)
from trt.packing import (
    MultimodalPromptProcessor,
    PackedLanguageInputs,
    PromptPackingSpec,
    PromptTensorInputs,
)
from trt.vision import GROOTVisualEmbed
from trt.language import (
    compile_groot_lm_trt_with_plugin,
    run_groot_plugin_language,
)
from trt.measure import (
    compare_full_groot_to_eager_actions,
    compare_groot_action_step,
    compare_vision,
    tensor_error_metrics,
)
from trt.plugin_utils import (
    register_plugin_op,
    load_plugin,
    patch_vision_attention,  
    restore_attention,
    infer_siglip_seq_len,
)

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    #"use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}

MODEL_ID = "nvidia/GR00T-N1.5-3B"
SEED = 42

GROOT_EMBODIMENT_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}

def make_compile_inputs(action_step, vl_embs, state, embodiment_id, device):
    batch_size = vl_embs.shape[0]
    dtype = vl_embs.dtype

    action_horizon = action_step.action_horizon
    action_dim = action_step.action_decoder.layer2.b.shape[-1]

    actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )

    timestep = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.long,
    )

    return (
        actions,
        timestep,
        vl_embs,
        state,
        embodiment_id,
    )

@torch.no_grad()
def build_groot_language_inputs(core, vit_embs, input_ids, attention_mask=None) -> PackedLanguageInputs:
    eagle = core.backbone.eagle_model
    image_token_index = getattr(
        eagle,
        "image_token_index",
        eagle.config.image_token_index,
    )

    processor = MultimodalPromptProcessor(
        PromptPackingSpec(
            style="chat_template_placeholder",
            token_embed_fn=eagle.language_model.get_input_embeddings(),
            image_token_id=image_token_index,
        )
    )

    return processor(
        PromptTensorInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embs=vit_embs,
        )
    )

@torch.no_grad()
def build_groot_context_inputs(core, vit_embs, input_ids, attention_mask):
    eagle = core.backbone.eagle_model
    packed = build_groot_language_inputs(
        core,
        vit_embs,
        input_ids,
        attention_mask,
    )

    out = eagle.language_model(
        inputs_embeds=packed.inputs_embeds,
        attention_mask=packed.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    context_embs = out.hidden_states[core.backbone.select_layer]
    context_embs = core.backbone.eagle_linear(context_embs)

    # Match action_head.process_backbone_output().
    vlln_weight = getattr(core.action_head.vlln, "weight", None)
    if vlln_weight is not None:
        context_embs = context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
    context_embs = core.action_head.vlln(context_embs)
    context_embs = core.action_head.vl_self_attention(context_embs)

    return (
        context_embs,
        packed.pad_mask,
        packed.attention_mask,
        packed.position_ids,
    )

def make_groot_context_masks(context_embs, attention_mask):
    context_pad_masks = attention_mask.to(device=context_embs.device, dtype=torch.bool)
    context_position_ids = torch.cumsum(context_pad_masks, dim=1) - 1

    return compact_prefix_inputs(
        context_embs,
        context_pad_masks,
        context_position_ids,
    )

@torch.no_grad()
def compare_groot_context(eager_context_embs, trt_context_embs, attention_mask, name="groot context"):

    compact_eager, _, _, _ = make_groot_context_masks(eager_context_embs, attention_mask)
    compact_trt, _, _, _ = make_groot_context_masks(trt_context_embs, attention_mask)

    tensor_error_metrics(name, compact_trt, compact_eager)

def _select_context_rows(context_embs, row_mask):
    row_mask = row_mask.to(device=context_embs.device, dtype=torch.bool)
    return torch.cat(
        [context_embs[b, row_mask[b], :] for b in range(context_embs.shape[0])],
        dim=0,
    )

@torch.no_grad()
def compare_groot_context_token_types(core, eager_context_embs, trt_context_embs, input_ids, attention_mask, name):
    eagle = core.backbone.eagle_model
    image_token_index = getattr(eagle, "image_token_index", eagle.config.image_token_index)

    valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    image_tokens = (input_ids == image_token_index) & valid
    text_tokens = (input_ids != image_token_index) & valid

    if int(image_tokens.sum().item()) > 0:
        tensor_error_metrics(
            f"{name} image tokens",
            _select_context_rows(trt_context_embs, image_tokens),
            _select_context_rows(eager_context_embs, image_tokens),
        )

    if int(text_tokens.sum().item()) > 0:
        tensor_error_metrics(
            f"{name} text tokens",
            _select_context_rows(trt_context_embs, text_tokens),
            _select_context_rows(eager_context_embs, text_tokens),
        )

def save_groot_visual_engine_for_edge_llm(
    model,
    pixel_values,
    engine_dir,
    *,
    device="cuda",
    dtype=torch.float16,
    model_type="groot_vision",
):
    engine_dir = pathlib.Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / "visual.engine"
    config_path = engine_dir / "config.json"

    pixel_values = pixel_values.to(device=device, dtype=dtype).contiguous()
    visual = GROOTVisualEmbed(model).eval().to(device=device, dtype=dtype)

    vision_model = model.backbone.eagle_model.vision_model.vision_model

    with torch.no_grad():
        eager_output = visual(pixel_values)

    batch_size, seq_len = infer_siglip_seq_len(vision_model, pixel_values)

    patched = []
    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )

    exported = torch.export.export(
        visual,
        args=(pixel_values,),
        strict=False,
    )

    input_spec = torch_tensorrt.Input.from_tensor(pixel_values)

    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        inputs=(input_spec,),
        disable_tf32=True,
        use_explicit_typing=True,
        use_fp32_acc=True,
        truncate_double=True,
        immutable_weights=True,
        decompose_attention=True,
        require_full_compilation=True,
    )
    
    if patched:
        restore_attention(patched)

    engine_path.write_bytes(engine_bytes)

    def tensor_meta(t):
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }

    output_meta = (
        [tensor_meta(t) for t in eager_output]
        if isinstance(eager_output, (tuple, list))
        else [tensor_meta(eager_output)]
    )

    config = {
        "model_type": model_type,
        "component": "vision",
        "engine_file": "visual.engine",
        "precision": "FP16",
        "input_names": ["pixel_values"],
        "inputs": {
            "pixel_values": tensor_meta(pixel_values),
        },
        "outputs": output_meta,
        "siglip_batch_size": batch_size,
        "siglip_seq_len": seq_len,
    }

    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return engine_path

def main() -> int:
    # Put every model and tensor on CUDA when available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained GROOT policy.
    policy = load_policy(GrootPolicy, MODEL_ID, device).to(device).eval()
    # The core model owns the vision, language/context, and action modules used below.
    model = policy._groot_model.to(device).eval()

    data, messages = load_test_data(
        dataset_id="lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    # create processor and get tokenizer ----
    cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
    processor = get_processor(str(cache_dir), 
        {
            'trust_remote_code': True, 
            'fix_mistral_regex': False
        })
    # ------
    
    model_inputs = prepare_model_inputs(
        processor,
        processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            },
        },
        data,
        messages,
        device,
    )

    tokenized_data = model_inputs['tokenized_data']
    input_ids = tokenized_data['input_ids']

    # groot specifc inputs ------
    attention_mask = tokenized_data['attention_mask']
    state, state_mask = pack_state(
        data["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    embodiment_id = torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

    # Keep the raw image pixels as a one-stream list so this mirrors the PI0.5 script.
    images = [tokenized_data["pixel_values"].to(
        device=device,
        dtype=torch.float16,
    )]
    pixel_values = images[0]
    # groot specifc inputs ------

    # Load the custom TensorRT plugin library before compiling plugin-backed modules.
    register_plugin_op()
    load_plugin()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    engine_dir = "/tmp/groot_edge_llm/visual"
    save_groot_visual_engine_for_edge_llm(
        model,
        pixel_values,
        engine_dir,
        device=device,
        dtype=torch.float16,
        model_type="groot_vision",
    )    
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())