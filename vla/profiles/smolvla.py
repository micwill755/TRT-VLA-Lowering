"""SmolVLA Edge-LLM compile profile."""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lerobot.policies.smolvla import SmolVLAPolicy

from trt.data import prepare_policy_batch
from trt.edge_llm_runtime import run_llm_inference_runtime_smoke
from trt.export.smolvla import SerializedSmolVLAVision, SmolVLAExportHooks
from trt.export.settings import ACTION_TRT_SETTINGS, VISION_TRT_SETTINGS
from trt.inference.smolvla import (
    SmolVLAInferenceHooks,
    run_inference_pytorch_smolvla,
    run_inference_smolvla_engines,
)
from trt.io_spec import PI05_EDGE_IO
from trt.measure import compute_action_parity_metrics
from trt.serialize import SerializedPI05Action, SerializedPI05Language
from trt.utils import load_policy

from vla.profile import InMemoryHandles, SerializedHandles, SerializedStageSpec, VLAProfile


class SmolVLAProfile(VLAProfile):
    name = "smolvla"
    model_id = "lerobot/smolvla_base"
    engine_dir_default = "/tmp/smolvla_edge_llm"
    display_name = "PyTorch SmolVLA"

    policy_cls = SmolVLAPolicy
    io = PI05_EDGE_IO
    fill_missing_cameras = True
    prefer_same_iter_reference = True

    serialized_stages = (
        SerializedStageSpec("vision", "visual", SerializedSmolVLAVision),
        SerializedStageSpec("language", "language", SerializedPI05Language),
        SerializedStageSpec("action", "action", SerializedPI05Action),
    )

    vision_trt_settings = dict(VISION_TRT_SETTINGS)
    action_trt_settings = dict(ACTION_TRT_SETTINGS)

    def load_policy(self, model_id: str, device: torch.device) -> tuple[Any, nn.Module]:
        policy = load_policy(self.policy_cls, model_id, device).to(device).eval()
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

    def get_tokenizer(self, *, policy: Any, args: argparse.Namespace) -> Any:
        del args
        return AutoTokenizer.from_pretrained(policy.model.config.vlm_model_name)

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> SmolVLAExportHooks:
        del args
        return SmolVLAExportHooks(
            io=self.io,
            tokenizer=tokenizer,
            vision_trt_settings=self.vision_trt_settings,
            action_trt_settings=self.action_trt_settings,
            max_generate_length=0,
        )

    def make_inference_hooks(self) -> SmolVLAInferenceHooks:
        return SmolVLAInferenceHooks()

    def post_export(
        self,
        runner: Any,
        engine_root: str | None,
    ) -> int | None:
        args = runner.args
        if not args.run_cpp_smoke or engine_root is None:
            return None

        smoke_input = pathlib.Path(engine_root) / "runtime_smoke" / "input.json"
        if not smoke_input.exists():
            raise FileNotFoundError(f"Missing runtime smoke input: {smoke_input}")

        print(f"\nRunning C++ llm_inference smoke: {smoke_input}")
        result = run_llm_inference_runtime_smoke(
            engine_root=args.engine_dir,
            input_file=smoke_input,
            llm_inference_bin=args.llm_inference_bin,
            max_generate_length=0,
            dump_output=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=os.sys.stderr)
        if result.returncode != 0:
            print(f"C++ smoke failed with exit code {result.returncode}")
            return result.returncode
        print("C++ smoke completed successfully.")
        return None

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
