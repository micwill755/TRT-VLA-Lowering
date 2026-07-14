import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch_tensorrt
from trt.utils import configure_thor_pytorch, force_hf_attention

configure_thor_pytorch()
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.factory import make_pre_post_processors
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.data import load_test_data, frame_from_test_data
from trt.modules.export.vision import GridVisionExportModule

def load_config():
    config = PI05Config(
        device="cpu",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=32,
        max_action_dim=32,
        image_resolution=(224, 224),
        input_features={
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
    )
    config.validate_features()
    return config, PI05Policy(config).eval()

device = torch.device("cuda")
dtype = torch.float16
print("1 load_plugins", flush=True)
load_plugins_for_trt()
print("2 load_config", flush=True)
config, policy = load_config()
print("3 model.to gpu fp16", flush=True)
model = policy.model.to(device=device, dtype=dtype).eval()
paligemma = model.paligemma_with_expert.paligemma.model
vision = paligemma.vision_tower
print("4 make_pre_post_processors", flush=True)
pre_processor, post_processor = make_pre_post_processors(
    config, None,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
print("5 load_test_data", flush=True)
data = load_test_data("lerobot/libero", episode_index=0, frame_index=0)
print("6 frame_from_test_data", flush=True)
frame = frame_from_test_data(data, policy, fill_missing=True)
print("7 pre_processor", flush=True)
model_inputs = pre_processor(frame)
print("8 preprocess_images", flush=True)
images, img_masks = policy._preprocess_images(model_inputs)
pixel_values = torch.cat([img.to(device=device, dtype=dtype) for img in images], dim=0).contiguous()
vision = vision.float()
force_hf_attention(vision, "eager")
print("9 GridVisionExportModule", flush=True)
visual = GridVisionExportModule(
    vision_model=vision,
    projector=paligemma.multi_modal_projector,
    sample_pixel_values=pixel_values.float(),
    select_layer=-1,
    pixel_shuffle=False,
    downsample_ratio=0.5,
    force_float32_input=True,
    vision_kwargs={},
).eval().to(device=device)
print("10 visual forward", flush=True)
with torch.no_grad():
    embs = visual(pixel_values)
print("OK", embs.shape, flush=True)
