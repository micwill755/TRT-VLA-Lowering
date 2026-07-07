"""PI0.5 Edge-LLM compile profile."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoTokenizer

from lerobot.policies.pi05 import PI05Policy

from trt.data import prepare_policy_batch
from trt.profile import VLAProfile
from trt.utils import force_hf_attention

PALIGEMMA_TOKENIZER_ID = "google/paligemma-3b-pt-224"


class Pi05Profile(VLAProfile):
    name = "pi05"
    model_id = "lerobot/pi05_libero"
    engine_dir_default = "/tmp/pi05_edge_llm"
    display_name = "PyTorch PI0.5"

    policy_cls = PI05Policy
    fill_missing_cameras = True

    def _init_policy(self) -> None:
        self.policy = self.policy_cls.from_pretrained(self.model_id).to(self.device).eval()
        self.config = self.policy.config

    def _init_models(self) -> None:
        self.model = self.policy.model.to(device=self.device, dtype=torch.float16).eval()

        paligemma = self.model.paligemma_with_expert.paligemma.model
        self.vision = paligemma.vision_tower
        self.lm = paligemma.language_model

        force_hf_attention(self.vision, "eager")
        force_hf_attention(self.lm, "eager")
        force_hf_attention(
            self.model.paligemma_with_expert.gemma_expert.model, "eager"
        )

    def _init_tokenizers(self) -> None:
        self.text_tok = AutoTokenizer.from_pretrained(PALIGEMMA_TOKENIZER_ID)
        self.text_tokenizer = self.text_tok

    def prepare_compile_inputs(
        self,
        *,
        data: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        return prepare_policy_batch(
            self.policy,
            self.pre_processor,
            data,
            self.device,
            args.model_id,
            fill_missing=self.fill_missing_cameras,
        )
