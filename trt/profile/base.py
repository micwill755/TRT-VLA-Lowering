from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn

from trt.io_spec import PipelineIOSpec
from trt.profile.handles import InMemoryHandles, SerializedHandles


class VLAProfile(ABC):
    """Owns HF/LeRobot setup for one VLA run: policy, core model, and compile inputs."""

    name: ClassVar[str] = "vla"
    pipeline_model_type: ClassVar[str] = ""
    model_id: ClassVar[str] = ""
    engine_dir_default: ClassVar[str] = "/tmp/vla_edge_llm"
    display_name: ClassVar[str] = "VLA"

    policy_cls: ClassVar[type | None] = None
    io: ClassVar[PipelineIOSpec | None] = None

    def __init__(self, device: torch.device, model_id: str | None = None) -> None:
        self.device = device
        self.model_id = model_id or type(self).model_id
        self.policy: Any = None
        self.model: nn.Module | None = None
        self.text_tok: Any = None
        self._init_policy()
        self._init_models()
        self._init_tokenizers()

    @abstractmethod
    def _init_policy(self) -> None:
        """Load ``self.policy`` from ``self.model_id``."""

    @abstractmethod
    def _init_models(self) -> None:
        """Set ``self.model`` to the export/inference core module."""

    def _init_tokenizers(self) -> None:
        """Optional: set ``self.text_tok`` for export sidecars."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        del parser

    @abstractmethod
    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Build ``model_inputs`` passed to export and benchmark."""

    def get_tokenizer(self, *, policy: Any = None, args: argparse.Namespace | None = None) -> Any:
        return self.text_tok

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> Any:
        raise NotImplementedError(f"{self.name} legacy export hooks are not configured")

    def post_export(self, ctx: Any, engine_root: str | None = None) -> int | None:
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
        raise NotImplementedError(f"{self.name} has no registered eager inference runner")

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
        raise NotImplementedError(f"{self.name} has no registered TRT inference runner")

