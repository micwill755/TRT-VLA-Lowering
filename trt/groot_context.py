"""GROOT language post-projection (hidden states -> context_embs)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from trt.compile import save_trt_engine_module
from trt.language import run_vla_lm_prefill


class GROOTContextProjectionWrapper(nn.Module):
    def __init__(self, eagle_linear, vlln, vl_self_attention):
        super().__init__()
        self.eagle_linear = eagle_linear
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, hidden_states: torch.Tensor):
        context_embs = self.eagle_linear(hidden_states)

        vlln_weight = getattr(self.vlln, "weight", None)
        if vlln_weight is not None:
            context_embs = context_embs.to(dtype=vlln_weight.dtype)

        context_embs = self.vlln(context_embs)
        context_embs = self.vl_self_attention(context_embs)
        return context_embs


def save_groot_language_post_engine(
    core,
    hidden_states: torch.Tensor,
    engine_dir,
    *,
    device,
    dtype=torch.float16,
    model_type: str = "language_post",
):
    projection = make_groot_context_projection(core, device=device, dtype=dtype)
    sample = hidden_states.to(device=device, dtype=dtype).contiguous()

    with torch.no_grad():
        example_output = projection(sample)

    return save_trt_engine_module(
        projection,
        (sample,),
        engine_dir,
        engine_file="language_post.engine",
        model_type=model_type,
        component="language_post",
        input_names=["hidden_states"],
        output_names=["context_embs"],
        example_output=example_output,
        extra_config={
            "hidden_size": int(sample.shape[-1]),
            "max_seq_len": int(sample.shape[1]),
            "batch_size": int(sample.shape[0]),
        },
    )


def make_groot_context_projection(core, *, device, dtype=torch.float16) -> GROOTContextProjectionWrapper:
    return GROOTContextProjectionWrapper(
        copy.deepcopy(core.backbone.eagle_linear).to(device=device, dtype=dtype).eval(),
        copy.deepcopy(core.action_head.vlln).to(device=device, dtype=dtype).eval(),
        copy.deepcopy(core.action_head.vl_self_attention).to(device=device, dtype=dtype).eval(),
    )


class GROOTLanguageAdapter:
    """Chain LLM hidden-state prefill with GR00T context projection."""

    def __init__(
        self,
        llm,
        projection: GROOTContextProjectionWrapper,
        *,
        max_kv_capacity: int,
        rope_cache: torch.Tensor,
        config,
        include_logits: bool = True,
    ):
        self.llm = llm
        self.projection = projection
        self.max_kv_capacity = int(max_kv_capacity)
        self.rope_cache = rope_cache
        self.config = config
        self.include_logits = bool(include_logits)

    def __call__(self, inputs_embeds: torch.Tensor, kv_caches=None, ctx_len=None):
        del kv_caches, ctx_len
        if hasattr(self.llm, "rope_cache"):
            _, hidden, _ = run_vla_lm_prefill(
                self.llm,
                inputs_embeds,
                max_kv_capacity=self.max_kv_capacity,
                rope_cache=self.rope_cache,
                config=self.config,
            )
        else:
            from trt.language import _llm_runner_sample_args, prefill_inputs

            inputs = prefill_inputs(
                inputs_embeds,
                self.config,
                max_kv_capacity=self.max_kv_capacity,
                rope_cache=self.rope_cache,
            )
            outputs = self.llm(*_llm_runner_sample_args(inputs))
            hidden = outputs[1 if self.include_logits else 0]
        return self.projection(hidden)
