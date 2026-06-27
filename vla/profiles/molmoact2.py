"""MolmoAct2 Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.molmoact2 import MolmoAct2Policy

from trt.data import prepare_policy_batch
from trt.export import MolmoAct2ExportHooks
from trt.export.molmoact2 import (
    SerializedMolmoAct2Action,
    SerializedMolmoAct2Backbone,
    action_output_dim,
)
from trt.export.molmoact2_pipeline import MolmoAct2ExportPipeline
from trt.export.settings import ACTION_TRT_SETTINGS
from trt.inference import MolmoAct2InferenceHooks
from trt.inference.molmoact2 import (
    run_inference_molmoact2_engines,
    run_inference_pytorch_molmoact2,
    run_inference_trt_molmoact2,
)
from trt.io_spec import MOLMOACT2_EDGE_IO
from trt.measure import compute_action_parity_metrics

from vla.profile import InMemoryHandles, SerializedHandles, SerializedStageSpec, VLAProfile


class MolmoAct2Profile(VLAProfile):
    name = "molmoact2"
    model_id = "allenai/MolmoAct2"
    engine_dir_default = "/tmp/molmoact2_edge_llm"
    display_name = "PyTorch MolmoAct2"

    policy_cls = MolmoAct2Policy
    io = MOLMOACT2_EDGE_IO
    uses_export_pipeline = True
    fill_missing_cameras = False
    prefer_same_iter_reference = True
    in_memory_trt_stage = "language"
    serialized_benchmark_stage = "language"

    serialized_stages = (
        SerializedStageSpec("language", "language", SerializedMolmoAct2Backbone),
        SerializedStageSpec("action", "action", SerializedMolmoAct2Action),
    )
    plugin_info_aliases = {
        "prefix_seq_len": ("action", "prefix_seq_len"),
        "chunk_size": ("action", "chunk_size"),
        "max_action_dim": ("action", "max_action_dim"),
        "num_inference_steps": ("action", "num_inference_steps"),
    }

    action_trt_settings = dict(ACTION_TRT_SETTINGS)

    def on_run_start(self, device: torch.device, args: argparse.Namespace) -> None:
        del device, args

    def load_policy(self, model_id: str, device: torch.device) -> tuple[Any, nn.Module]:
        from trt.utils import load_policy

        policy = load_policy(self.policy_cls, model_id, device).to(device).eval()
        if policy.config.inference_action_mode is None:
            policy.config.inference_action_mode = "continuous"
        model = policy.model.to(device).eval()
        return policy, model

    def prepare_compile_inputs(
        self,
        *,
        policy: Any,
        data: dict[str, Any],
        device: torch.device,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        return prepare_policy_batch(
            policy,
            data,
            device,
            args.model_id,
            fill_missing=self.fill_missing_cameras,
        )

    def export_pipeline_cls(self):
        return MolmoAct2ExportPipeline

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> MolmoAct2ExportHooks:
        del tokenizer
        return MolmoAct2ExportHooks(
            io=self.io,
            action_trt_settings=self.action_trt_settings,
            stage_parity=not args.no_stage_parity,
        )

    def make_inference_hooks(self) -> MolmoAct2InferenceHooks:
        return MolmoAct2InferenceHooks()

    def finalize_serialized_handles(self, handles: SerializedHandles, policy: Any) -> None:
        handles.plugin_info.setdefault("output_action_dim", action_output_dim(policy))

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
                plugin_info=handles.plugin_info,
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
            plugin_info=handles.plugin_info,
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
