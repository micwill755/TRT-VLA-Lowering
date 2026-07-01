"""MolmoAct2 Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.molmoact2 import MolmoAct2Policy

from trt.data import prepare_policy_batch
from trt.export.molmoact2 import MolmoAct2ExportHooks
from trt.export.settings import ACTION_TRT_SETTINGS
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

    policy_cls = MolmoAct2Policy
    io = MOLMOACT2_EDGE_IO
    fill_missing_cameras = False

    action_trt_settings = dict(ACTION_TRT_SETTINGS)

    def _init_policy(self) -> None:
        self.policy = self.policy_cls.from_pretrained(self.model_id).to(self.device).eval()
        if self.policy.config.inference_action_mode is None:
            self.policy.config.inference_action_mode = "continuous"

    def _init_models(self) -> None:
        self.model = self.policy.model.to(self.device).eval()
        self.vision = self.model.model.vision_backbone

    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        return prepare_policy_batch(
            self.policy,
            data,
            self.device,
            args.model_id,
            fill_missing=self.fill_missing_cameras,
        )

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> MolmoAct2ExportHooks:
        del tokenizer, args
        return MolmoAct2ExportHooks(
            io=self.io,
            action_trt_settings=self.action_trt_settings,
        )

    def run_inference_eager(
        self,
        model: nn.Module,
        policy: Any,
        compile_inputs: dict[str, Any],
        *,
        seed: int,
        device: torch.device,
        vision_module=None,
    ) -> tuple[torch.Tensor, dict, float]:
        del model, vision_module
        return run_inference_pytorch_molmoact2(
            model,
            policy,
            compile_inputs,
            seed=seed,
            device=device,
            io=self.io,
        )

    def run_inference_trt(
        self,
        model: nn.Module,
        policy: Any,
        compile_inputs: dict[str, Any],
        *,
        handles: InMemoryHandles | SerializedHandles,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict, float]:
        if isinstance(handles, SerializedHandles):
            return run_inference_molmoact2_engines(
                model,
                policy,
                compile_inputs,
                backbone_runner=handles.language,
                diffusion_runner=handles.action,
                seed=seed,
                device=device,
                io=self.io,
            )
        return run_inference_trt_molmoact2(
            model,
            policy,
            compile_inputs,
            trt_backbone=handles.language,
            trt_diffusion=handles.action,
            seed=seed,
            device=device,
            io=self.io,
        )

    def compute_action_metrics(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        policy: Any,
    ) -> dict[str, float]:
        from trt.export.molmoact2 import crop_policy_actions

        return compute_action_parity_metrics(
            crop_policy_actions(policy, pred_actions),
            crop_policy_actions(policy, target_actions),
        )
