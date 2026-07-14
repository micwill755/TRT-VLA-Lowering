import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

config = PI05Config(
    device="cpu", chunk_size=50, n_action_steps=50,
    max_state_dim=32, max_action_dim=32, image_resolution=(224, 224),
    input_features={
        f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
    },
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
)
config.validate_features()
policy = PI05Policy(config).eval()
vision = policy.model.paligemma_with_expert.paligemma.model.vision_tower

device = torch.device("cuda")
print("moving vision to gpu", flush=True)
vision = vision.to(device=device, dtype=torch.float16).eval()
x = torch.randn(2, 3, 224, 224, device=device, dtype=torch.float16)
print("forward fp16", flush=True)
with torch.no_grad():
    out = vision(pixel_values=x.float(), return_dict=True)
print("ok fp16", out.last_hidden_state.shape, flush=True)

x32 = x.float()
print("forward fp32", flush=True)
with torch.no_grad():
    out2 = vision(pixel_values=x32, return_dict=True)
print("ok fp32", out2.last_hidden_state.shape, flush=True)
