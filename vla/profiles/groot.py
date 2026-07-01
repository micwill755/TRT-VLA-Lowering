"""GR00T Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.groot_n1 import DEFAULT_TOKENIZER_ASSETS_REPO
from lerobot.utils.constants import HF_LEROBOT_HOME

from trt.data import create_pil_messages, prepare_model_inputs
from trt.export.groot import GrootExportHooks
from trt.export.settings import ACTION_TRT_SETTINGS, VISION_TRT_SETTINGS
from trt.helper import get_processor
from trt.io_spec import GROOT_EDGE_IO
from trt.profile import VLAProfile


class GrootProfile(VLAProfile):
    name = "groot"
    pipeline_model_type = "Gr00tN1d7"
    model_id = "nvidia/GR00T-N1.5-3B"
    engine_dir_default = "/tmp/groot_edge_llm"
    display_name = "PyTorch GR00T"

    policy_cls = GrootPolicy
    io = GROOT_EDGE_IO

    vision_trt_settings = dict(VISION_TRT_SETTINGS)
    action_trt_settings = dict(ACTION_TRT_SETTINGS)

    def _init_policy(self) -> None:
        self.policy = self.policy_cls.from_pretrained(self.model_id).to(self.device).eval()

    def _init_models(self) -> None:
        self.model = self.policy._groot_model.to(self.device).eval()

    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        del args
        pil_messages = create_pil_messages(data)
        cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
        processor = get_processor(
            str(cache_dir),
            {
                "trust_remote_code": True,
                "fix_mistral_regex": False,
            },
        )
        self.text_tok = getattr(processor, "tokenizer", processor)
        return prepare_model_inputs(
            processor,
            processor.process_vision_info,
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
            self.device,
        )

    def get_tokenizer(self, *, policy: Any = None, args: argparse.Namespace | None = None) -> Any:
        del policy, args
        if self.text_tok is None:
            raise RuntimeError("prepare_compile_inputs must run before get_tokenizer")
        return self.text_tok

    def make_export_hooks(self, *, tokenizer: Any, args: argparse.Namespace) -> GrootExportHooks:
        del args
        return GrootExportHooks(
            io=self.io,
            tokenizer=tokenizer,
            vision_trt_settings=self.vision_trt_settings,
            action_trt_settings=self.action_trt_settings,
        )

