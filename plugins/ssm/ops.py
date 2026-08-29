"""Custom ops for Edge-LLM SSM plugins (causal conv1d, Mamba SSD, Gated DeltaNet).

Signatures follow ``tensorrt_edgellm/models/ops.py``. Eager bodies are shape
stubs; Dynamo uses ``register_fake``. The converter turns these into TRT plugin
nodes at compile time.
"""

from __future__ import annotations

from typing import Tuple

import torch

from plugins.common import has_torch_op


def register_ops() -> None:
    _register_causal_conv1d()
    _register_update_ssm_state()
    _register_gated_delta_net()


def _register_causal_conv1d() -> None:
    if has_torch_op("trt_edgellm", "causal_conv1d"):
        return

    @torch.library.custom_op("trt_edgellm::causal_conv1d", mutates_args=())
    def causal_conv1d(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        conv_state: torch.Tensor,
        context_lengths: torch.Tensor,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del weight, bias, context_lengths, stride, padding, dilation, groups
        return torch.zeros_like(hidden_states), conv_state.clone()

    @causal_conv1d.register_fake
    def _(hidden_states, weight, bias, conv_state, context_lengths, stride, padding, dilation, groups):
        del weight, bias, context_lengths, stride, padding, dilation, groups
        return torch.empty_like(hidden_states), torch.empty_like(conv_state)


def _register_update_ssm_state() -> None:
    if has_torch_op("trt_edgellm", "update_ssm_state"):
        return

    @torch.library.custom_op("trt_edgellm::update_ssm_state", mutates_args=())
    def update_ssm_state(
        hidden_states: torch.Tensor,
        ssm_a: torch.Tensor,
        ssm_b: torch.Tensor,
        ssm_c: torch.Tensor,
        ssm_d: torch.Tensor,
        dt: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        context_lengths: torch.Tensor,
        dt_softplus: int,
        ngroups: int,
        nheads: int,
        head_dim: int,
        dstate: int,
        chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, context_lengths
        del dt_softplus, ngroups, nheads, head_dim, dstate, chunk_size
        return torch.zeros_like(hidden_states), state.clone()

    @update_ssm_state.register_fake
    def _(
        hidden_states,
        ssm_a,
        ssm_b,
        ssm_c,
        ssm_d,
        dt,
        dt_bias,
        state,
        context_lengths,
        dt_softplus,
        ngroups,
        nheads,
        head_dim,
        dstate,
        chunk_size=0,
    ):
        del ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, context_lengths
        del dt_softplus, ngroups, nheads, head_dim, dstate, chunk_size
        return torch.empty_like(hidden_states), torch.empty_like(state)


def _register_gated_delta_net() -> None:
    if has_torch_op("trt_edgellm", "gated_delta_net"):
        return

    @torch.library.custom_op("trt_edgellm::gated_delta_net", mutates_args=())
    def gated_delta_net(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        h0_source: torch.Tensor,
        context_lengths: torch.Tensor,
        k_dim: int,
        v_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del q, k, a, b, A_log, dt_bias, context_lengths, k_dim, v_dim
        return torch.zeros_like(v), h0_source.clone()

    @gated_delta_net.register_fake
    def _(q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths, k_dim, v_dim):
        del q, k, a, b, A_log, dt_bias, context_lengths, k_dim, v_dim
        return torch.empty_like(v), torch.empty_like(h0_source)
