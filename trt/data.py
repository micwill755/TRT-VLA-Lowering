"""Dataset loading and model-specific input preparation for VLA Edge-LLM export.

Shared entry point is ``load_test_data`` (raw LeRobot frame). Each policy family
then runs its own prepare step:

- PI0.5 / SmolVLA: ``prepare_policy_batch`` (LeRobot preprocessor → token batch)
- GR00T: ``create_pil_messages`` + ``prepare_model_inputs`` (Eagle processor)
- GROOT action head: ``pack_state`` (pad state to ``max_state_dim``)
"""

import torch

from PIL import Image

from collections.abc import Callable
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

from trt import helper

# Camera keys read from a LeRobot dataset frame (libero provides image + image2).
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_DATASET_ID = "lerobot/libero"

def frame_from_test_data(
    data: dict[str, Any],
    policy,
    *,
    fill_missing: bool = False,
) -> dict[str, Any]:
    """Flatten ``load_test_data`` output into a LeRobot observation dict.

    ``load_test_data`` nests cameras under ``data["images"]``; the policy
    preprocessor expects flat keys like ``observation.images.image`` alongside
    ``observation.state`` and ``task``. When ``fill_missing=True``, zero tensors
    are inserted for any camera keys declared in ``policy.config.input_features``
    but absent from the dataset frame (PI0.5 / SmolVLA multi-camera configs).
    """
    frame = dict(data["images"])
    frame[OBS_STATE] = data["state"]
    frame["task"] = data.get("task", "")

    if fill_missing:
        for key, feature in policy.config.input_features.items():
            if key.startswith("observation.images.") and key not in frame:
                frame[key] = torch.zeros(feature.shape, dtype=torch.float32)

    return frame

def load_test_data(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    episode_index: int = 0,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Load one raw observation from a LeRobot dataset (no policy preprocessing).

    Returns a model-agnostic dict:

    - ``images``: ``{observation.images.*: float32 [3, H, W] in [0, 1]}``
    - ``state``: proprio vector from the frame
    - ``task``: language instruction string (defaults to ``"Perform the task."``)

    Downstream prepare functions (``prepare_policy_batch``, ``create_pil_messages``,
    etc.) convert this into model-specific tensors.
    """
    dataset = LeRobotDataset(dataset_id, episodes=[episode_index])
    frame = dataset[frame_index]

    images = {
        key: frame[key] for key in IMAGE_KEYS if key in frame
    }

    return {
        "images": images,
        "state": frame[OBS_STATE],
        "task": frame.get("task", "") or "Perform the task.",
    }


def prepare_policy_batch(
    policy,
    pre_processor,
    data: dict[str, Any],
    device: str | torch.device,
    model_id: str,
    *,
    fill_missing: bool = False,
) -> dict[str, Any]:
    """Run the LeRobot preprocessor and return a PI0.5 / SmolVLA model batch."""
    del model_id
    frame = frame_from_test_data(data, policy, fill_missing=fill_missing)
    model_inputs = pre_processor(frame)
    return helper.to_device(model_inputs, device)


# ------------------ GROOT SPECIFIC ------------------ 

def create_pil_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build HF chat messages with PIL images for Eagle-style processors (GR00T).

    Eagle's ``apply_chat_template`` / ``process_vision_info`` expect message content
    like ``{"type": "image", "image": <PIL.Image>}``, not raw CHW tensors.
    Images are sorted by key so camera order is stable across runs.
    """
    images = data["images"]
    task = str(data.get("task", "") or "Perform the task.")

    image_content = [
        {"type": "image", "image": _tensor_image_to_pil(img)}
        for _, img in sorted(images.items())
    ]

    return [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": str([task])}],
        }
    ]

def prepare_model_inputs(
    processor,
    vision_info_fn: Callable,
    chat_args: dict[str, Any],
    processor_args: dict[str, Any],
    data: dict[str, Any],
    messages: list[dict[str, Any]],
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Tokenize chat messages + images via the Eagle (or similar) HF processor.

    Applies the chat template, extracts vision inputs from ``messages``, and runs
    the processor to produce ``input_ids``, ``attention_mask``, and
    ``pixel_values``. Passes through ``state`` and ``task`` from ``data`` for the
    action head. Used by GR00T export after ``create_pil_messages``.
    """
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
        "task": data["task"],
    }

    # TODO: remove this function
    return helper.to_device(model_inputs, device)

def pack_state(
    state: torch.Tensor,
    max_state_dim: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Pad or truncate a LeRobot state vector to GROOT's fixed action-head width.

    Raw dataset frames often provide a small state vector shaped (D,). GROOT's
    state encoder is trained with input_dim=max_state_dim and consumes a
    sequence-like state tensor shaped (B, state_horizon, max_state_dim). For a
    single frame, state_horizon is 1, so we add batch/history axes and pad or
    truncate the feature dimension to the fixed model width.
    """
    state = torch.as_tensor(state, dtype=torch.float32, device=device)

    # libero often starts as (D,), so add batch + horizon axes
    if state.ndim == 1:
        state = state.unsqueeze(0)

    if state.ndim == 2:
        state = state.unsqueeze(1)

    bsz, _, state_dim = state.shape

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

    return state

def _tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert a LeRobot CHW float tensor in [0, 1] to an RGB PIL image."""
    img = img.detach().cpu()

    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)

    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)

    return Image.fromarray(img.numpy())

# ------------------ GROOT SPECIFIC ------------------ 