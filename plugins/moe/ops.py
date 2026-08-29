"""Custom op for Edge-LLM ``Fp16MoePlugin``.

NVFP4 / INT4 MoE plugins use the same converter pattern with extra scale
tensors; this example stays on the unquantized CuTe DSL grouped-GEMM plugin.
"""

from __future__ import annotations

import torch

from plugins.common import has_torch_op


def register_ops() -> None:
    if has_torch_op("trt_edgellm", "Fp16MoePlugin"):
        return

    @torch.library.custom_op("trt_edgellm::Fp16MoePlugin", mutates_args=())
    def fp16_moe_plugin(
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_weights: torch.Tensor,
        fc2_weights: torch.Tensor,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_inter_size: int,
        activation_type: int,
        norm_topk_prob: int,
        max_routed_rows: int,
    ) -> torch.Tensor:
        del router_logits, fc1_weights, fc2_weights
        del num_experts, top_k, hidden_size, moe_inter_size
        del activation_type, norm_topk_prob, max_routed_rows
        return torch.zeros_like(hidden_states)

    @fp16_moe_plugin.register_fake
    def _(
        router_logits,
        hidden_states,
        fc1_weights,
        fc2_weights,
        num_experts,
        top_k,
        hidden_size,
        moe_inter_size,
        activation_type,
        norm_topk_prob,
        max_routed_rows,
    ):
        del router_logits, fc1_weights, fc2_weights
        del num_experts, top_k, hidden_size, moe_inter_size
        del activation_type, norm_topk_prob, max_routed_rows
        return torch.empty_like(hidden_states)
