from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn

from lerobot.policies.factory import make_pre_post_processors

from trt.utils import find_pack_step
from trt.io_spec import PipelineIOSpec
from trt.profile.handles import InMemoryHandles, SerializedHandles
from trt.data import frame_from_test_data

class VLAProfile(ABC):
    """Owns HF/LeRobot setup for one VLA run: policy, core model, and compile inputs."""

    name: ClassVar[str] = "vla"
    pipeline_model_type: ClassVar[str] = ""
    model_id: ClassVar[str] = ""
    engine_dir_default: ClassVar[str] = "/tmp/vla_edge_llm"
    display_name: ClassVar[str] = "VLA"

    io: ClassVar[PipelineIOSpec | None] = None

    def __init__(self, device: torch.device, model_id: str | None = None) -> None:
        self.device = device
        self.model_id = model_id or type(self).model_id
        self.policy: Any = None
        self.model: nn.Module | None = None
        self.pre_processor: Any = None
        self.post_processor: Any = None
        self.text_tok: Any = None
        self.action_tok: Any = None
        self._init_policy()
        self._init_models()
        self._init_processors()
        self._init_tokenizers()

    @abstractmethod
    def _init_policy(self) -> None:
        """Load ``self.policy`` from ``self.model_id``."""

    @abstractmethod
    def _init_models(self) -> None:
        """Set ``self.model`` to the export/inference core module."""

    def _init_processors(self) -> None:
        """Build LeRobot pre/post processor pipelines from ``self.policy.config``."""
        self.pre_processor, self.post_processor = make_pre_post_processors(
            self.config,
            None,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def _init_tokenizers(self) -> None:
        pack = find_pack_step(self.pre_processor)
        if pack is not None:
            self.text_tok = getattr(pack.processor, "tokenizer", None)
            self.action_tok = getattr(pack, "action_processor", None) or getattr(
                self.policy, "action_tokenizer", None
            )
            return

        self.text_tok = getattr(self.pre_processor, "tokenizer", None)

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