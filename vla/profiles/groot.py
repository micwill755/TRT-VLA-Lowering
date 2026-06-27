"""GR00T Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.groot_n1 import DEFAULT_TOKENIZER_ASSETS_REPO
from lerobot.utils.constants import HF_LEROBOT_HOME

from trt.data import create_pil_messages, prepare_model_inputs
from trt.export import GrootExportHooks
from trt.export.settings import ACTION_TRT_SETTINGS, VISION_TRT_SETTINGS
from trt.helper import get_processor
from trt.inference import (
    GrootInferenceHooks,
    compute_groot_policy_action_metrics,
    run_inference_pytorch_groot,
    run_inference_trt_plugin,
)
from trt.io_spec import GROOT_EDGE_IO
from trt.serialize import (
    SerializedGrootAction,
    SerializedGrootActionContext,
    SerializedGrootLanguage,
    SerializedGrootVision,
)
from trt.utils import load_policy

from vla.profile import InMemoryHandles, SerializedHandles, SerializedStageSpec, VLAProfile

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROOT_RUNTIME_BIN = (
    WORKSPACE_ROOT
    / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/groot/groot_runtime_smoke"
)

class GrootProfile(VLAProfile):
    name = "groot"
    model_id = "nvidia/GR00T-N1.5-3B"
    engine_dir_default = "/tmp/groot_edge_llm"
    display_name = "PyTorch GR00T"

    policy_cls = GrootPolicy
    io = GROOT_EDGE_IO
    uses_export_pipeline = True

    serialized_stages = (
        SerializedStageSpec("vision", "visual", SerializedGrootVision),
        SerializedStageSpec("language", "language", SerializedGrootLanguage),
        SerializedStageSpec("action_context", "action_context", SerializedGrootActionContext),
        SerializedStageSpec("action", "action", SerializedGrootAction),
    )
    plugin_info_aliases = {
        "language_max_seq_len": ("language", "max_seq_len"),
        "context_seq_len": ("action", "context_seq_len"),
        "context_hidden_size": ("action", "context_hidden_size"),
    }

    vision_trt_settings = dict(VISION_TRT_SETTINGS)
    action_trt_settings = dict(ACTION_TRT_SETTINGS)
    cpp_smoke_bin = DEFAULT_GROOT_RUNTIME_BIN

    def __init__(self) -> None:
        self._processor: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--groot-runtime-bin",
            type=str,
            default=str(DEFAULT_GROOT_RUNTIME_BIN),
            help="Path to the C++ groot_runtime_smoke executable.",
        )
        parser.add_argument(
            "--vision-engine-dir",
            type=str,
            default=None,
            help="Optional override for the vision engine directory.",
        )
        parser.add_argument(
            "--language-engine-dir",
            type=str,
            default=None,
            help="Optional override for the language engine directory.",
        )
        parser.add_argument(
            "--skip-language",
            action="store_true",
            help="Skip language.engine export.",
        )
        parser.add_argument(
            "--skip-action",
            action="store_true",
            help="Skip action/diffusion engine export.",
        )

    def load_policy(self, model_id: str, device: torch.device) -> tuple[Any, nn.Module]:
        policy = load_policy(self.policy_cls, model_id, device).to(device).eval()
        model = policy._model.to(device).eval()
        return policy, model

    def prepare_compile_inputs(
        self,
        *,
        policy: Any,
        data: dict[str, Any],
        device: torch.device,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        del policy
        pil_messages = create_pil_messages(data)
        cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
        self._processor = get_processor(
            str(cache_dir),
            {
                "trust_remote_code": True,
                "fix_mistral_regex": False,
            },
        )
        return prepare_model_inputs(
            self._processor,
            self._processor.process_vision_info,
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
            device,
        )

    def get_tokenizer(self, *, policy: Any, args: argparse.Namespace) -> Any:
        del policy, args
        if self._processor is None:
            raise RuntimeError("prepare_compile_inputs must run before get_tokenizer")
        return getattr(self._processor, "tokenizer", self._processor)

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> GrootExportHooks:
        del args
        return GrootExportHooks(
            io=self.io,
            tokenizer=tokenizer,
            vision_trt_settings=self.vision_trt_settings,
            action_trt_settings=self.action_trt_settings
        )

    def make_inference_hooks(self) -> GrootInferenceHooks:
        return GrootInferenceHooks()

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
        return run_inference_pytorch_groot(
            model,
            policy,
            compile_inputs,
            seed=seed,
            device=device,
            vision_module=vision_module,
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
        action_context = getattr(handles, "action_context", None)
        return run_inference_trt_plugin(
            model,
            policy,
            compile_inputs,
            trt_vision=handles.vision,
            trt_lm=handles.language,
            trt_diffusion=handles.action,
            trt_action_context=action_context,
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
        return compute_groot_policy_action_metrics(pred_actions, target_actions, policy)
