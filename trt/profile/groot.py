"""GR00T Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1 import DEFAULT_TOKENIZER_ASSETS_REPO
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE

from trt.data import create_pil_messages, prepare_model_inputs
from trt.modules.export.diffusion import DEFAULT_DIFFUSION_TRT_SETTINGS as ACTION_TRT_SETTINGS
from trt.vision import DEFAULT_VISION_TRT_SETTINGS as VISION_TRT_SETTINGS
from trt.helper import get_processor
from trt.io_spec import GROOT_EDGE_IO
from trt.profile import VLAProfile
from trt.utils import force_hf_attention

class GrootProfile(VLAProfile):
    name = "groot"
    pipeline_model_type = "Gr00tN1d7"
    model_id = "nvidia/GR00T-N1.5-3B"
    engine_dir_default = "/tmp/groot_edge_llm"
    display_name = "PyTorch GR00T"

    policy_cls = GrootPolicy
    io = GROOT_EDGE_IO

    vision_trt_settings = dict(VISION_TRT_SETTINGS)
    action_trt_settings = dict(ACTION_TRT_SETTINGS)

    def _init_policy(self) -> None:
        self.config = GrootConfig(
            base_model_path=self.model_id,
            device=str(self.device),
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
                # GROOT predicts action chunks, not a single 7-d vector
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(50, 32)),
            },
        )
        self.policy = GrootPolicy(self.config).to(self.device).eval()

    def _init_models(self) -> None:
        self.model = self.policy._groot_model
        self.lm = self.model.backbone.eagle_model.language_model
        self.vision = self.model.backbone.eagle_model.vision_model
        self.action = self.model.action_head

        force_hf_attention(self.vision, "eager")
        force_hf_attention(self.lm, "eager")

    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        pil_messages = create_pil_messages(data)
        cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
        processor = get_processor(
            str(cache_dir),
            {
                "trust_remote_code": True,
                "fix_mistral_regex": False,
            },
        )
        self.text_tok = getattr(processor, "tokenizer", processor)
        return prepare_model_inputs(
            processor,
            processor.process_vision_info,
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

    def get_tokenizer(self, *, policy: Any = None, args: argparse.Namespace | None = None) -> Any:
        if self.text_tok is None:
            raise RuntimeError("prepare_compile_inputs must run before get_tokenizer")
        return self.text_tok