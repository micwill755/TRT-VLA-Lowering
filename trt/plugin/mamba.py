"""Nemotron Mamba mixer wrapper and ``torch.ops.trt`` custom ops.

Mirrors ``attention.py`` / ``plugin_utils._register_attention_plugin_op``:
eager stubs + fake kernels for Dynamo, lowered by ``plugin_converter`` onto
Edge-LLM ``causal_conv1d`` and ``update_ssm_state`` IPluginV3.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DT_CLAMP = 50.0


def _has_torch_op(namespace: str, name: str) -> bool:
    return hasattr(torch.ops, namespace) and hasattr(getattr(torch.ops, namespace), name)


def register_mamba_plugin_ops() -> None:
    """Register ``trt::causal_conv1d`` and ``trt::update_ssm_state``."""
    if _has_torch_op("trt", "causal_conv1d") and _has_torch_op("trt", "update_ssm_state"):
        return

    @torch.library.custom_op("trt::causal_conv1d", mutates_args=())
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
    def _(
        hidden_states,
        weight,
        bias,
        conv_state,
        context_lengths,
        stride,
        padding,
        dilation,
        groups,
    ):
        del weight, bias, context_lengths, stride, padding, dilation, groups
        return torch.empty_like(hidden_states), torch.empty_like(conv_state)

    @torch.library.custom_op("trt::update_ssm_state", mutates_args=())
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, context_lengths
        del dt_softplus, ngroups, nheads, head_dim, dstate
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
    ):
        del ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, context_lengths
        del dt_softplus, ngroups, nheads, head_dim, dstate
        return torch.empty_like(hidden_states), torch.empty_like(state)


class PluginNemotronMamba(nn.Module):
    """Wrap ``NemotronHMamba2Mixer``: native GEMMs, plugin conv + SSM.

    Export contract (not HF ``cache_params``)::

        hidden, conv_state, ssm_state, context_lengths
            -> hidden, conv_state_out, ssm_state_out
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.in_proj = original.in_proj
        self.out_proj = original.out_proj
        self.conv1d = original.conv1d
        self.norm = original.norm
        self.A_log = original.A_log
        self.D = original.D
        self.dt_bias = original.dt_bias

        self.num_heads = int(original.num_heads)
        self.head_dim = int(original.head_dim)
        self.n_groups = int(original.n_groups)
        self.ssm_state_size = int(original.ssm_state_size)
        self.conv_dim = int(original.conv_dim)
        self.conv_kernel = int(
            getattr(original, "conv_kernel_size", original.conv1d.kernel_size[0])
        )
        self.layer_idx = getattr(original, "layer_idx", None)
        self._group_size = (self.num_heads * self.head_dim) // self.n_groups
        self._eps = float(
            getattr(original.norm, "variance_epsilon", getattr(original.norm, "eps", 1e-5))
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        d_inner = self.num_heads * self.head_dim
        d_state = self.n_groups * self.ssm_state_size

        projected = self.in_proj(hidden_states)
        gate, conv_in, dt = projected.split(
            [d_inner, self.conv_dim, self.num_heads], dim=-1
        )
        dt = dt.clamp(-_DT_CLAMP, _DT_CLAMP)

        conv_bias = self.conv1d.bias
        if conv_bias is None:
            conv_bias = torch.zeros(
                self.conv_dim, device=conv_in.device, dtype=conv_in.dtype
            )

        conv_out, conv_state_out = torch.ops.trt.causal_conv1d.default(
            conv_in,
            self.conv1d.weight,
            conv_bias,
            conv_state,
            context_lengths,
            1,
            self.conv_kernel - 1,
            1,
            self.conv_dim,
        )
        conv_out = F.silu(conv_out)

        ssm_input, ssm_b, ssm_c = conv_out.split([d_inner, d_state, d_state], dim=-1)
        ssm_input = ssm_input.view(batch_size, seq_len, self.num_heads, self.head_dim)
        ssm_b = ssm_b.view(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        ssm_c = ssm_c.view(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        ssm_a = -torch.exp(self.A_log.to(torch.float32))
        ssm_out, ssm_state_out = torch.ops.trt.update_ssm_state.default(
            ssm_input,
            ssm_a,
            ssm_b,
            ssm_c,
            self.D.to(torch.float16),
            dt,
            self.dt_bias.to(torch.float16),
            ssm_state,
            context_lengths,
            1,
            self.n_groups,
            self.num_heads,
            self.head_dim,
            self.ssm_state_size,
        )
        ssm_out = ssm_out.view(batch_size, seq_len, d_inner)
        return self.out_proj(self._gated_rmsnorm(ssm_out, gate)), conv_state_out, ssm_state_out

    def _gated_rmsnorm(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = (hidden_states * F.silu(gate)).to(torch.float32)
        grouped = gated.view(*gated.shape[:-1], -1, self._group_size)
        variance = (grouped * grouped).mean(-1, keepdim=True)
        normed = grouped * torch.rsqrt(variance + self._eps)
        return normed.view(*hidden_states.shape).to(torch.float16) * self.norm.weight
