"""Per-VLA registration contract for Edge-LLM compile scripts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

import argparse
import torch
import torch.nn as nn

from trt.io_spec import PipelineIOSpec
from trt.serialize import SerializedModuleSpec

if TYPE_CHECKING:
    from trt.export.hooks import VLAExportHooks
    from trt.inference.hooks import VLAInferenceHooks
    from vla.base_compile_edge_llm import BaseEdgeCompileRunner


@dataclass(frozen=True)
class SerializedStageSpec:
    """One serialized TRT stage under ``engine_root/<engine_subdir>/``."""

    key: str
    engine_subdir: str
    wrapper_cls: type
    optional: bool = False

    def to_module_spec(self) -> SerializedModuleSpec:
        return SerializedModuleSpec(self.key, self.engine_subdir, self.wrapper_cls)


@dataclass
class InMemoryHandles:
    vision: Any = None
    language: Any = None
    action: Any = None
    action_context: Any = None


@dataclass
class SerializedHandles:
    vision: Any = None
    language: Any = None
    action_context: Any = None
    action: Any = None


class VLAProfile(ABC):
    """Model-specific hooks + metadata consumed by ``BaseEdgeCompileRunner``."""

    name: ClassVar[str] = "vla"
    model_id: ClassVar[str] = ""
    engine_dir_default: ClassVar[str] = "/tmp/vla_edge_llm"
    display_name: ClassVar[str] = "PyTorch VLA"

    policy_cls: ClassVar[type]
    io: ClassVar[PipelineIOSpec]

    uses_export_pipeline: ClassVar[bool] = True
    fill_missing_cameras: ClassVar[bool] = False

    serialized_stages: ClassVar[tuple[SerializedStageSpec, ...]] = ()

    vision_trt_settings: ClassVar[dict] = {}
    action_trt_settings: ClassVar[dict] = {}
    use_fused_in_memory_language: ClassVar[bool] = False

    cpp_smoke_bin: ClassVar[Path | None] = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register model-specific CLI flags."""

    def on_run_start(self, device: torch.device, args: argparse.Namespace) -> None:
        del device, args

    @abstractmethod
    def prepare_compile_inputs(
        self,
        *,
        policy: Any,
        data: dict[str, Any],
        device: torch.device,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Build inputs passed to export/inference."""

    def get_tokenizer(self, *, policy: Any, args: argparse.Namespace) -> Any:
        del policy, args
        return None

    def make_export_hooks(
        self,
        *,
        tokenizer: Any,
        args: argparse.Namespace,
    ) -> VLAExportHooks:
        del tokenizer, args
        raise NotImplementedError(f"{self.name} does not use VLAExportPipeline")

    def make_inference_hooks(self) -> VLAInferenceHooks | None:
        return None

    def export_pipeline_cls(self):
        from trt.export.pipeline import VLAExportPipeline

        return VLAExportPipeline

    def run_export_phase(
        self,
        runner: BaseEdgeCompileRunner,
    ) -> tuple[InMemoryHandles, str | None]:
        del runner
        raise NotImplementedError(f"{self.name} export is not implemented")

    def post_export(
        self,
        runner: BaseEdgeCompileRunner,
        engine_root: str | None,
    ) -> int | None:
        """Optional C++ smoke or early exit. Return an exit code to stop the runner."""
        del runner, engine_root
        return None

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def compute_action_metrics(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        policy: Any,
    ) -> dict[str, float]:
        ...
