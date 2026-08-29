"""Qwen3-MoE-style block: gate GEMM stays native TRT, experts go through the plugin."""

from __future__ import annotations

import torch
import torch.nn as nn


def pack_swiglu_fc1(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """64-row up/gate interleave expected by ``Fp16MoePlugin`` SwiGLU FC1."""
    inter, hidden = up.shape
    chunk = 64
    if inter % chunk != 0:
        raise ValueError(f"moe_inter_size ({inter}) must be a multiple of {chunk}")
    n_chunks = inter // chunk
    up_chunks = up.reshape(n_chunks, chunk, hidden)
    gate_chunks = gate.reshape(n_chunks, chunk, hidden)
    return torch.stack([up_chunks, gate_chunks], dim=1).reshape(2 * inter, hidden)


class PluginFp16MoE(nn.Module):
    """Router linear + fused softmax/topk/grouped-GEMM plugin.

    Plugin constraints (from ``fp16MoePlugin.cpp``): ``num_experts`` in
    {128, 256}, ``hidden_size % 128 == 0``, ``moe_inter_size % 64 == 0``,
    ``activation_type`` 2 (SwiGLU) or 4 (ReLU2).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        moe_inter_size: int = 64,
        num_experts: int = 128,
        top_k: int = 2,
        activation_type: int = 2,
        norm_topk_prob: int = 1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.moe_inter_size = int(moe_inter_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.activation_type = int(activation_type)
        self.norm_topk_prob = int(norm_topk_prob)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        chunk = 64
        n_chunks = moe_inter_size // chunk
        up = torch.randn(num_experts, moe_inter_size, hidden_size) * 0.02
        gate_w = torch.randn(num_experts, moe_inter_size, hidden_size) * 0.02
        down = torch.randn(num_experts, hidden_size, moe_inter_size) * 0.02
        up_chunks = up.reshape(num_experts, n_chunks, chunk, hidden_size)
        gate_chunks = gate_w.reshape(num_experts, n_chunks, chunk, hidden_size)
        fc1 = torch.stack([up_chunks, gate_chunks], dim=2).reshape(
            num_experts, 2 * moe_inter_size, hidden_size
        )
        self.register_buffer("fc1_weights", fc1)
        self.register_buffer("fc2_weights", down)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape
        router_logits = self.gate(hidden_states.reshape(-1, hidden)).float()
        max_routed_rows = batch * seq_len * self.top_k
        return torch.ops.trt_edgellm.Fp16MoePlugin.default(
            router_logits,
            hidden_states,
            self.fc1_weights,
            self.fc2_weights,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_inter_size,
            self.activation_type,
            self.norm_topk_prob,
            max_routed_rows,
        )
