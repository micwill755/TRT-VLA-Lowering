import torch

from PIL import Image

from collections.abc import Callable
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

from trt import helper

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_DATASET_ID = "lerobot/libero"

# TODO: deprecate this function
def make_batch(policy, model_id, device, fill_missing=False, dataset_id=DEFAULT_DATASET_ID, episode_index=0, frame_index=0):
    preprocess, _ = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Load only first episode
    dataset = LeRobotDataset(dataset_id, episodes=[episode_index])
    current_frame = dataset[frame_index]

    # Extract images and state from the current frame
    current_images = {
        key: current_frame[key]
        for key in IMAGE_KEYS
        if key in current_frame
    }

    # frame = raw LeRobot observation dict with images, state, and task (if available)
    frame = dict(current_images)
    frame[OBS_STATE] = current_frame[OBS_STATE]
    frame["task"] = current_frame.get("task", "")

    # Ensure all expected image keys are present, filling in zeros for any missing ones.
    if fill_missing:
        for key, feature in policy.config.input_features.items():
            if key.startswith("observation.images.") and key not in frame:
                frame[key] = torch.zeros(feature.shape, dtype=torch.float32)

    return preprocess(frame)

def _tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Convert a LeRobot image tensor into a PIL image.

    GROOT's Eagle processor expects chat message image entries like:
        {"type": "image", "image": <PIL.Image.Image>}
    rather than raw CHW torch tensors.
    """
    img = img.detach().cpu()

    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)

    # LeRobot images are usually CHW.
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)

    return Image.fromarray(img.numpy())

def load_test_data(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    episode_index: int = 0,
    frame_index: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = LeRobotDataset(dataset_id, episodes=[episode_index])
    frame = dataset[frame_index]

    images = {
        key: frame[key] for key in IMAGE_KEYS if key in frame
    }

    data = {
        "images": images,
        "state": frame[OBS_STATE],
        "task": frame.get("task", "") or "Perform the task.",
    }

    image_content = [
        {"type": "image", "image": _tensor_image_to_pil(img)}
        for _, img in sorted(images.items())
    ]

    messages = [
        {
            "role": "user",
            "content": image_content
            + [{"type": "text", "text": str([data["task"]])}],
        }
    ]

    return data, messages

# prepare model inputs will be different per model using processor
def prepare_model_inputs(
    processor,
    vision_info_fn: Callable,
    chat_args: dict[str, Any],
    processor_args: dict[str, Any],
    data: dict[str, Any],
    messages: list[dict[str, Any]],
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        **chat_args,
    )

    image_inputs, video_inputs = vision_info_fn(messages)

    tokenized_data = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **processor_args,
    )

    model_inputs = {
        "tokenized_data": tokenized_data,
        "state": data["state"],
        "task": data["task"]
    }

    return helper.to_device(model_inputs, device)

def pack_state(
    state: torch.Tensor,
    max_state_dim: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(state, dtype=torch.float32, device=device)

    # Dataset gives (D,), GROOT wants batched state.
    if state.ndim == 1:
        state = state.unsqueeze(0)

    # GROOT action head wants (B, 1, max_state_dim).
    if state.ndim == 2:
        state = state.unsqueeze(1)

    bsz, _, state_dim = state.shape
    used_dim = min(state_dim, max_state_dim)

    if state_dim > max_state_dim:
        state = state[:, :, :max_state_dim]
    elif state_dim < max_state_dim:
        pad = torch.zeros(
            bsz,
            1,
            max_state_dim - state_dim,
            dtype=state.dtype,
            device=device,
        )
        state = torch.cat([state, pad], dim=-1)

    state_mask = torch.zeros(
        bsz,
        1,
        max_state_dim,
        dtype=torch.bool,
        device=device,
    )
    state_mask[:, :, :used_dim] = True

    return state, state_mask