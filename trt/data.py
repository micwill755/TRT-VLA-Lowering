import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_DATASET_ID = "lerobot/libero"

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
