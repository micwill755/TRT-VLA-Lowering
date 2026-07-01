"""SmolVLA Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lerobot.policies.smolvla import SmolVLAPolicy

from trt.data import prepare_policy_batch
from trt.export.smolvla import SmolVLAExportHooks
from trt.modules.export.diffusion import DEFAULT_DIFFUSION_TRT_SETTINGS as ACTION_TRT_SETTINGS
from trt.vision import DEFAULT_VISION_TRT_SETTINGS as VISION_TRT_SETTINGS
from trt.inference.smolvla import (
    run_inference_pytorch_smolvla,
    run_inference_smolvla_engines,
)

from trt.io_spec import PI05_EDGE_IO
from trt.measure import compute_action_parity_metrics
from trt.profile import InMemoryHandles, SerializedHandles, VLAProfile


class SmolVLAProfile(VLAProfile):
    name = "smolvla"
    model_id = "lerobot/smolvla_base"
    engine_dir_default = "/tmp/smolvla_edge_llm"
    display_name = "PyTorch SmolVLA"

    policy_cls = SmolVLAPolicy
    io = PI05_EDGE_IO
    fill_missing_cameras = True

    vision_trt_settings = dict(VISION_TRT_SETTINGS)
    action_trt_settings = dict(ACTION_TRT_SETTINGS)

    def _init_policy(self) -> None:
        self.policy = self.policy_cls.from_pretrained(self.model_id).to(self.device).eval()

    def _init_models(self) -> None:
        self.model = self.policy.model.to(self.device).eval()

    def _processor_pretrained_path(self) -> str | None:
        return self.model_id

    def _init_tokenizers(self) -> None:
        super()._init_tokenizers()
        if self.text_tok is None:
            self.text_tok = AutoTokenizer.from_pretrained(self.policy.model.config.vlm_model_name)

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

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> SmolVLAExportHooks:
        del args
        return SmolVLAExportHooks(
            io=self.io,
            tokenizer=tokenizer,
            vision_trt_settings=self.vision_trt_settings,
            action_trt_settings=self.action_trt_settings,
            max_generate_length=0,
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
        del vision_module
        return run_inference_pytorch_smolvla(
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
        if handles.vision is None:
            raise RuntimeError("SmolVLA serialized inference requires loaded vision handles")
        return run_inference_smolvla_engines(
            model,
            policy,
            compile_inputs,
            vision_runner=handles.vision,
            language_runner=handles.language,
            diffusion_runner=handles.action,
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
        del policy
        return compute_action_parity_metrics(pred_actions, target_actions)
