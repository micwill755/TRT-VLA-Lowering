import torch

from lerobot.policies.pi05 import PI05Policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

def make_batch(policy, model_id, device):
    preprocess, _ = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    episode_index = 0
    frame_index = 0

    dataset = LeRobotDataset("lerobot/libero", episodes=[episode_index])
    current_frame = dataset[frame_index]

    # Extract images and state from the current frame
    current_images = {
        key: current_frame[key]
        for key in IMAGE_KEYS
        if key in current_frame
    }

    # frame = raw LeRobot observation dict with images, state, and task (if available)
    frame = dict(current_images)
    frame["observation.state"] = current_frame[STATE_KEY]
    frame["task"] = current_frame.get("task", "")

    # Ensure all expected image keys are present, filling in zeros for any missing ones.
    for key, feature in policy.config.input_features.items():
        if key.startswith("observation.images.") and key not in frame:
            frame[key] = torch.zeros(feature.shape, dtype=torch.float32)

    return preprocess(frame)


def load_policy(model_id, device):
    policy = PI05Policy.from_pretrained(model_id, device=device)
    return policy.to(device).eval()

def prepare_policy_inputs(policy, batch):
    # Extract the relevant inputs for the policy from the batch
    device = next(policy.parameters()).device
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
    return images, img_masks, tokens, masks

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/pi05_base"
    policy = load_policy(model_id, device)
    batch = make_batch(policy, model_id, device)
    core = policy.core

    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch)

    with torch.no_grad():
        # Run the policy core to ensure it works before compilation
        _ = core(
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
        )

if __name__ == "__main__":
    main()