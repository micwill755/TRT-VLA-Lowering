"""SmolVLA Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from trt.data import prepare_policy_batch
from trt.io_spec import PI05_EDGE_IO
from trt.profile import VLAProfile
from trt.utils import force_hf_attention


class SmolVLAProfile(VLAProfile):
    name = "smolvla"
    model_id = "lerobot/smolvla_base"
    engine_dir_default = "/tmp/smolvla_edge_llm"
    display_name = "PyTorch SmolVLA"

    io = PI05_EDGE_IO
    fill_missing_cameras = True

    action_trt_settings = {
        "disable_tf32": True,
        "use_explicit_typing": True,
        "truncate_double": True,
        "immutable_weights": True,
        "decompose_attention": True,
        "require_full_compilation": True,
        "offload_module_to_cpu": True,
        "use_fp32_acc": True,
    }

    def _init_policy(self) -> None:
        self.config = SmolVLAConfig(
            device=str(self.device),
            chunk_size=50,
            n_action_steps=50,
            max_state_dim=32,
            max_action_dim=32,
            resize_imgs_with_padding=(512, 512),
            load_vlm_weights=True,
            vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            input_features={
                f"{OBS_IMAGES}.image": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 224, 224)
                ),
                f"{OBS_IMAGES}.image2": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 224, 224)
                ),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
            },
        )
        self.config.validate_features()
        self.policy = SmolVLAPolicy(self.config).to(self.device).eval()

    def _init_models(self) -> None:
        self.model = self.policy.model.to(device=self.device, dtype=torch.float16).eval()

        vlm = self.model.vlm_with_expert.get_vlm_model()
        force_hf_attention(vlm.vision_model, "eager")
        force_hf_attention(vlm.text_model, "eager")
        force_hf_attention(self.model.vlm_with_expert.lm_expert, "eager")

    def _init_tokenizers(self) -> None:
        self.text_tok = self.policy.model.vlm_with_expert.processor.tokenizer
        self.text_tokenizer = self.text_tok

    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        return prepare_policy_batch(
            self.policy,
            self.pre_processor,
            data,
            self.device,
            args.model_id,
            fill_missing=self.fill_missing_cameras,
        )
