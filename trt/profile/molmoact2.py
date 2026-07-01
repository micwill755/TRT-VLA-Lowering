"""MolmoAct2 Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.molmoact2 import MolmoAct2Policy

from trt.data import prepare_policy_batch
from trt.export.molmoact2 import MolmoAct2ExportHooks
from trt.modules.export.diffusion import DEFAULT_DIFFUSION_TRT_SETTINGS as ACTION_TRT_SETTINGS
from trt.inference.molmoact2 import (
    run_inference_molmoact2_engines,
    run_inference_pytorch_molmoact2,
    run_inference_trt_molmoact2,
)
from trt.io_spec import MOLMOACT2_EDGE_IO
from trt.measure import compute_action_parity_metrics

from trt.profile import InMemoryHandles, SerializedHandles, VLAProfile


class MolmoAct2Profile(VLAProfile):
    name = "molmoact2"
    model_id = "allenai/MolmoAct2"
    engine_dir_default = "/tmp/molmoact2_edge_llm"
    display_name = "PyTorch MolmoAct2"
    
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

    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Build ``model_inputs`` passed to export and benchmark."""
        frame = frame_from_test_data(data)
        return self.pre_processor(frame)
