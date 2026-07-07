from __future__ import annotations

import torch

from trt.context import EdgeContext
from trt.data import pack_state
from trt.executor.models.groot.helpers import make_embodiment_id
from trt.data import (
    load_test_data,
    create_pil_messages,
    prepare_model_inputs,
    pack_state
)

from trt.plugin.plugin_utils import load_plugins_for_trt

def preprocess(ctx: EdgeContext) -> None:
    device = ctx.device

    load_plugins_for_trt()

    data = load_test_data(
        "lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    pil_messages = create_pil_messages(data)

    chat_args = {
        "add_generation_prompt": True
    }
    processor_args = {
        "images_kwargs": {
            "min_dynamic_tiles": 1,
            "max_dynamic_tiles": 1,
            "use_thumbnail": False,
        }
    }
    text = ctx.profile.eagle_processor.apply_chat_template(
        messages,
        tokenize=False,
        **chat_args,
    )

    image_inputs, video_inputs = ctx.profile.eagle_processor.process_vision_info(messages)

    tokenized_data = ctx.profile.eagle_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **processor_args,
    )

    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    pixel_values = tokenized_data["pixel_values"].to(device=device, dtype=ctx.dtype).contiguous()
        state = pack_state(
        data["state"],
        max_state_dim=64,
        device=device,
    ).to(device=device, dtype=ctx.dtype).contiguous()

    embodiment_id = make_embodiment_id(ctx.policy, state, device, torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "state": state,
        "embodiment_id": embodiment_id,
    }

def postprocess(ctx: EdgeContext) -> None:
    pass