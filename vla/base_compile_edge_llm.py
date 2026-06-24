"""Shared skeleton for VLA Edge-LLM export scripts.

Subclasses set class attributes (model id, default engine dir, etc.) and implement
model-specific hooks. The common LeRobot path is:

  load_test_data -> load_policy -> prepare_policy_batch -> export/benchmark
"""

from __future__ import annotations

import argparse
import pathlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch

from trt.data import load_test_data, prepare_policy_batch

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_ID = "lerobot/libero"
SEED = 42
DEFAULT_LLM_INFERENCE_BIN = (
    WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference"
)

class BaseEdgeBuilder(ABC):
    """Template-method base for Edge-LLM export scripts."""

    name: ClassVar[str] = "vla"
    model_id: ClassVar[str] = ""
    engine_dir_default: ClassVar[str] = "/tmp/vla_edge_llm"
    fill_missing_cameras: ClassVar[bool] = False

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        self.args = args or self.parse_args()
        self.device = torch.device(
            self.args.device if torch.cuda.is_available() else "cpu"
        )
        self.data: dict[str, Any] | None = None
        self.policy: Any = None
        self.model: Any = None
        self.compile_inputs: dict[str, Any] | None = None

    @classmethod
    def parse_args(cls, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"Export {cls.name} TensorRT engines for TensorRT-Edge-LLM",
        )
        parser.add_argument("--model-id", type=str, default=cls.model_id)
        parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
        parser.add_argument("--episode-index", type=int, default=0)
        parser.add_argument("--frame-index", type=int, default=0)
        parser.add_argument("--engine-dir", type=str, default=cls.engine_dir_default)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--llm-inference-bin",
            type=str,
            default=str(DEFAULT_LLM_INFERENCE_BIN),
            help="Path to TensorRT-Edge-LLM llm_inference binary for C++ smoke tests.",
        )
        parser.add_argument("--seed", type=int, default=SEED)
        parser.add_argument("--max-seq-len", type=int, default=None)
        parser.add_argument("--num-traj-samples", type=int, default=1)
        parser.add_argument("--max-generation-length", type=int, default=256)
        parser.add_argument(
            "--export-only",
            action="store_true",
            help="Export serialized .engine files; skip in-memory TRT plugin compile.",
        )
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--no-accuracy-check", action="store_true")
        parser.add_argument("--no-stage-parity", action="store_true")
        parser.add_argument("--run-cpp-smoke", action="store_true")
        parser.add_argument("--skip-export", action="store_true")
        parser.add_argument("--skip-pytorch", action="store_true")
        parser.add_argument("--skip-trt", action="store_true")
        parser.add_argument("--skip-engine", action="store_true")
        parser.add_argument("--num-iterations", type=int, default=12)
        parser.add_argument("--warmup", type=int, default=3)
        cls.add_arguments(parser)
        return parser.parse_args(argv)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Override to register model-specific CLI flags."""

    def load_data(self) -> dict[str, Any]:
        self.data = load_test_data(
            dataset_id=self.args.dataset_id,
            episode_index=self.args.episode_index,
            frame_index=self.args.frame_index,
        )
        return self.data

    @abstractmethod
    def load_policy(self) -> Any:
        """Load ``self.policy`` and ``self.model``."""

    def prepare_inputs(self) -> dict[str, Any]:
        if self.policy is None or self.data is None:
            raise RuntimeError("Call load_data() and load_policy() before prepare_inputs().")
        self.compile_inputs = prepare_policy_batch(
            self.policy,
            self.data,
            self.device,
            self.args.model_id,
            fill_missing=self.fill_missing_cameras,
        )
        return self.compile_inputs

    def run(self) -> int:
        self.load_data()
        self.load_policy()
        self.prepare_inputs()
        return self.export_and_benchmark()

    @abstractmethod
    def export_and_benchmark(self) -> int:
        """Compile/export engines and run parity/timing loops."""
