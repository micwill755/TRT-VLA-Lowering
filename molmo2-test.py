
import sys
import torch

from typing import Any 

from transformers import AutoModelForImageTextToText, AutoConfig
from transformers import AutoProcessor

from lerobot.policies.molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2 import MolmoAct2Config, MolmoAct2Policy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors

from trt.utils import (
    force_hf_attention, 
    find_pack_step
)

from trt.data import (
    load_test_data, 
    frame_from_test_data
)

from trt.plugin_utils import load_plugins_for_trt
from trt.profile import VLAProfile

class MolmoAct2Profile(VLAProfile):
    name = "molmo2"
    pipeline_model_type = "MolmoAct2"
    
    def _init_policy(self):
        self.config = MolmoAct2Config(
            checkpoint_path=self.model_id,
            device=str(self.device),
            inference_action_mode="continuous",
            # Optional: loads LIBERO norm stats + prompt/camera metadata from the checkpoint
            input_features={
                "observation.images.image": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 224, 224)
                ),
                "observation.images.wrist_image": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 224, 224)
                ),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            },
        )
        self.policy = MolmoAct2Policy(self.config)
    
    def _init_models(self):
        self.lm = self.policy.model.model.transformer
        self.vision = self.policy.model.model.vision_backbone
        self.action = self.policy.model.model.action_expert

        force_hf_attention(self.vision, "eager")
        force_hf_attention(self.lm, "eager")

    def _init_tokenizers(self):
        # The pipeline has no top-level .tokenizer. 
        # Walk its steps and find MolmoAct2PackInputsProcessorStep (registered as "molmoact2_pack_inputs")
        pack = find_pack_step(self.pre_processor)
        self.text_tok = pack.processor.tokenizer
        self.action_tok = pack.action_processor or getattr(policy, "action_tokenizer", None)

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_plugins_for_trt()
    
    profile = MolmoAct2Profile(
        model_id="allenai/MolmoAct2",
        device=device
    )

    export = VLExportEdgeLLMPipeline(
        profile=profile
    )
    export(dataset_id="lerobot/libero")

if __name__ == "__main__":
    SystemExit(main())