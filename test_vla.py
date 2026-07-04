import torch
import argparse
import torch_tensorrt
import logging

torch_tensorrt.logging.set_level(logging.ERROR)

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep

from trt.modules.export.vision import GridVisionExportModule
from trt.executor.models.groot.helpers import make_embodiment_id
from trt.data import create_pil_messages, prepare_model_inputs
from trt.utils import force_hf_attention
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.vision import nchw_to_hwc
from trt.data import (
    load_test_data, 
    frame_from_test_data,
    pack_state
)

from trt.plugin.plugin_utils import patch_vision_attention, patch_vision_attention_reference
from trt.compile import _make_input_spec

from typing import Any

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    #"use_fp32_acc": True,
    "truncate_double": True,
    #"use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
    "use_fp32_acc": True,
}

VISION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "use_fp32_acc": True,
}

def load_config(device):
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.5-3B",
        device=str(device),
        embodiment_tag="new_embodiment",  # or "gr1", "oxe_droid", etc.
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=64,
        max_action_dim=32,
        image_size=(224, 224),
        tokenizer_assets_repo="lerobot/eagle2hg-processor-groot-n1p5",
        # Match lerobot/libero camera keys (see Test/trt/data.py IMAGE_KEYS)
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

def prepare_compile_inputs(
    self,
    *,
    data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pil_messages = create_pil_messages(data)
    return prepare_model_inputs(
        self.eagle_processor,
        self.eagle_processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            }
        },
        data,
        pil_messages,
        self.device,
    )

# ----- LANGUAGE -------
def build_language_inputs(
        vision_model,
        image_embs,
        input_ids,
        attention_mask,
    ):
    pass


# ----- LANGUAGE -------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_plugins_for_trt()

    config, policy = load_config(device)
    model = policy._groot_model
    vision = model.backbone.eagle_model.vision_model
    force_hf_attention(vision, "eager")

    pre_processor, post_processor = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    # get tokenizer
    eagle_step = next(
        s for s in pre_processor.steps
        if isinstance(s, GrootEagleEncodeStep)
    )
    proc = eagle_step.proc
    eagle_processor = proc
    text_tok = getattr(proc, "tokenizer", proc)

    data = load_test_data(
        "lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    pil_messages = create_pil_messages(data)
    model_inputs = prepare_model_inputs(
        eagle_processor,
        eagle_processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            }
        },
        data,
        pil_messages,
        device,
    )
    
    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    pixel_values = tokenized_data["pixel_values"]
    state = pack_state(
        model_inputs["state"],  # [7] libero proprio
        max_state_dim=64,  # 64
        device=device,
    ) 
    state = state.to(device=device, dtype=torch.float16)
    action_side = {
        "state": state,
        "embodiment_id": make_embodiment_id(policy, state, device),
    }

    print(pixel_values.shape)
    pixel_values = pixel_values.to(device=device, dtype=torch.float16).contiguous()

    '''# ------ run eager on vision -----
    images_hwc = nchw_to_hwc(pixel_values)
    eagle = model.backbone.eagle_model
    visual = GridVisionExportModule(
        vision_model=vision,
        projector=eagle.mlp1,
        sample_pixel_values=images_hwc,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
        vision_kwargs={},
    ).eval().to(device=device, dtype=torch.float16)

    with torch.no_grad():
        image_embs = visual(pixel_values)

    print(image_embs.shape)
    # ------ run eager on vision -----

    # ------ run plugin on vision -----
    pixel_values_nchw = pixel_values.to(device=device, dtype=torch.float16).contiguous()
    hidden_states = vision.vision_model.embeddings(pixel_values_nchw)
    batch_size, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    patched = patch_vision_attention(
        vision.vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )

    sample_inputs = (pixel_values,)

    exported = torch.export.export(
        visual,
        args=sample_inputs,
        strict=False,
    )

    input_specs = _make_input_spec(sample_inputs)

    vision_settings = {
        **VISION_TRT_SETTINGS,
        "use_python_runtime": True,
    }
    trt_engine = torch_tensorrt.dynamo.compile(
        exported,
        inputs=input_specs,
        **vision_settings,
    )
    image_embs_trt_engine = trt_engine(pixel_values)
    #print(image_embs_trt_engine == image_embs)
    # ------ run plugin on vision -----
    print(torch.allclose(image_embs_trt_engine.float(), image_embs.float(), rtol=1e-2, atol=1e-2))'''

    # Align everything to one layout + dtype
    pixel_values = pixel_values.to(device=device, dtype=torch.float16).contiguous()

    eagle = model.backbone.eagle_model
    visual = GridVisionExportModule(
        vision_model=vision,
        projector=eagle.mlp1,
        sample_pixel_values=pixel_values,      # NCHW; module handles layout internally
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
        vision_kwargs={},
    ).eval().to(device=device, dtype=torch.float16)

    # --- Rung A: eager SDPA (UNPATCHED) ---
    with torch.no_grad():
        embs_eager = visual(pixel_values)

    # --- Patch SigLIP attention -> ViTPluginAttention ---
    hidden_states = vision.vision_model.embeddings(pixel_values)
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    patched = patch_vision_attention(
        vision.vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    # replace using self implemented siglip attention to compare
    '''
    patched = patch_vision_attention_reference(
        vision.vision_model
    )'''

    try:
        # --- Rung B: eager, but now with the plugin attention ---
        with torch.no_grad():
            embs_eager_plugin = visual(pixel_values)

        # --- Rung C: TRT compiled from the patched module ---
        exported = torch.export.export(visual, args=(pixel_values,), strict=False)
        input_specs = _make_input_spec((pixel_values,))
        trt_engine = torch_tensorrt.dynamo.compile(
            exported,
            inputs=input_specs,
            **{**VISION_TRT_SETTINGS, "use_python_runtime": True},
        )
        with torch.no_grad():
            embs_trt = trt_engine(pixel_values)
    finally:
        # always undo the patch so later eager runs aren't affected
        from trt.plugin.plugin_utils import restore_attention
        restore_attention(patched)

    # --- Localize the error ---
    parity("A vs C (prod)", embs_eager, embs_trt)
    # two below are not valid, 
    # b attention outputs will be zeros - it will run empty custom_op operations
    # c is running kernels producing correct attention outputs
    parity("A vs B (plugin)", embs_eager, embs_eager_plugin)
    #parity("B vs C (trt only)", embs_eager_plugin, embs_trt)

    # STEP 2 language
    print('Compiling language')
    build_language_inputs(
        eager,
        
    )

    return 0

def parity(name, a, b):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    rel_l2 = (a - b).norm() / b.norm().clamp_min(1e-8)
    rel_mean_pct = diff.mean() / b.abs().mean().clamp_min(1e-8) * 100
    close = torch.isclose(a, b, rtol=1e-2, atol=1e-2).float().mean() * 100
    print(
        f"{name:<22} mean_abs={diff.mean():.6f}  max_abs={diff.max():.6f}  "
        f"rel_l2={rel_l2:.4f}  rel_mean%={rel_mean_pct:.2f}  close%={close:.1f}"
    )
    
if __name__ == "__main__":
    SystemExit(main())