import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

'''
sample["observation.images.image"]   # camera view 1
sample["observation.images.image2"]  # camera view 2

when building frames, pull both camera images out of the dataset sample.
'''
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")

STATE_KEY = "observation.state"
ACTION_KEY = "action"


class VLATestSample:
    def __init__(self, input: dict[str, any], gt_actions: [], metadata: dict[str, any]):
        self.input = input
        self.gt_actions = gt_actions
        self.metadata = metadata


def compare_predictions(pred_action, gt_actions) -> bool:
    '''
    Compare the predicted action to the ground truth action(s).
    For simplicity, this example just checks if the predicted action is close to any of the future ground truth actions.
    In a real test, you might want a more sophisticated comparison depending on the action space and task.
    '''
    for gt_action in gt_actions:
        if torch.allclose(pred_action, gt_action, atol=1e-3):
            return True

    return False

def compare_single_action(pred_action: torch.Tensor, gt_actions: list[torch.Tensor]) -> dict[str, float]:
    """Compare one predicted 7D action against the matching ground-truth action."""
    pred = pred_action.squeeze(0).detach().cpu().float()
    gt = gt_actions[0].detach().cpu().float()

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

    diff = pred - gt
    abs_diff = diff.abs()

    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    print("Action comparison")
    print("-----------------")
    for name, p, g, d in zip(names, pred, gt, diff):
        print(
            f"{name:8s} pred={p.item(): .4f} "
            f"gt={g.item(): .4f} "
            f"diff={d.item(): .4f}"
        )

    metrics = {
        "mae": abs_diff.mean().item(),
        "mse": (diff ** 2).mean().item(),
        "l2": torch.linalg.norm(diff).item(),
        "xyz_l2": torch.linalg.norm(diff[:3]).item(),
        "rot_l2": torch.linalg.norm(diff[3:6]).item(),
        "gripper_abs": abs_diff[6].item(),
    }

    print("\nMetrics")
    print("-------")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    return metrics

def run_inference(model_id, policy, input_chunk, device='cuda'):
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    frame = to_lerobot_frame(input_chunk, policy)
    batch = preprocess(frame)

    with torch.inference_mode():
        pred_action = policy.select_action(batch)
        pred_action = postprocess(pred_action)

    return pred_action

def load_test_data(future_steps: int) -> VLATestSample:
    """Load one LIBERO sample and its future action targets for a VLA test."""

    if future_steps < 0:
        raise ValueError("future_steps must be non-negative")

    # For this test, always use the first episode and start at the first frame.
    episode_index = 0
    frame_index = 0

    # Load only the requested episode from the LeRobot dataset.
    dataset = LeRobotDataset("lerobot/libero", episodes=[episode_index])

    '''
    sample["observation.images.image"]   # image from camera 1 at this timestep
    sample["observation.images.image2"]  # image from camera 2 at this timestep
    sample["observation.state"]          # current robot state at this timestep
    sample["action"]                     # action he policy should output at this timestep
    '''

    # Current observation is the input frame the policy will condition on.
    current_frame = dataset[frame_index]

    # Collect the current frame plus the next `future_steps - 1` frames.
    # Change this range to start at `frame_index + 1` if you want strictly future frames.
    future_samples = [dataset[i] for i in range(frame_index, frame_index + future_steps)]

    # LeRobot stores camera images under flat observation keys.
    current_images = {key: current_frame[key] for key in IMAGE_KEYS if key in current_frame}

    # Package observations into the shape expected by VLATestSample.
    input_data = {
        "frames": current_images,
        "proprio": current_frame[STATE_KEY],
        "future_frames": [
            {key: sample[key] for key in IMAGE_KEYS if key in sample}
            for sample in future_samples
        ],
        "future_proprio": [
            sample[STATE_KEY]
            for sample in future_samples
        ],
        "task": current_frame.get("task"),
    }

    # Ground-truth actions are the target actions for each future sample.
    '''
    The usual convention is:

    state[0] + action[0] -> state[1] - at state 0 I take action 0 to get to state 1
    state[1] + action[1] -> state[2]
    state[2] + action[2] -> state[3]
    state[t] + action[t] -> state[t + 1]

    the ground truth action sequence for current frame t is:
    action[t], action[t + 1], action[t + 2], ...

    '''
    gt_actions = [sample[ACTION_KEY] for sample in future_samples]

    # Keep dataset bookkeeping so failures/debug output can be traced back.
    metadata = {
        "episode_index": current_frame["episode_index"],
        "frame_index": current_frame["frame_index"],
        "dataset_index": current_frame["index"],
        "future_steps": future_steps,
    }

    return VLATestSample(
        input=input_data,
        gt_actions=gt_actions,
        metadata=metadata,
    )

def to_lerobot_frame(input_data: dict, policy) -> dict:
    frame = dict(input_data["frames"])
    frame["observation.state"] = input_data["proprio"]
    frame["task"] = input_data.get("task", "")

    for key, feature in policy.config.input_features.items():
        if key.startswith("observation.images.") and key not in frame:
            frame[key] = torch.zeros(feature.shape, dtype=torch.float32)

    return frame

def load_policy(model_id, device):
    policy = PI05Policy.from_pretrained(model_id, device=device)
    policy = policy.to(device)
    return policy

#--- TRT -----

def compile_model_for_trt(policy, sample_input):
    # This is a placeholder function. The actual implementation would depend on the specifics of the model and the input.
    # You would need to trace the model with a representative input and then convert it to TensorRT format.
    # For example, you might use torch.jit.trace to create a TorchScript version of the model, and then use NVIDIA's tools to convert that to TensorRT.
    traced_model = torch.jit.trace(policy, sample_input)
    # Convert traced_model to TensorRT format here (this step is non-trivial and requires additional code).
    trt_model = convert_to_trt(traced_model)
    return trt_model

#--- TRT -----


def main() -> int:
    sample = load_test_data(5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_policy('lerobot/pi05_libero', device=device)
    # test since inference with the original PyTorch model works before trying to compile with TRT
    pred = run_inference('lerobot/pi05_libero',policy, sample.input, device=device)
    trt_model = compile_model_for_trt(policy, sample.input)

    return 0

    '''policy = PI05Policy.from_pretrained('lerobot/pi05_libero', device='cuda')
    dataset = LeRobotDataset("lerobot/libero")

    print (policy.config.input_features.keys())
    print(dataset[0].keys())'''

if __name__ == "__main__":
    main()