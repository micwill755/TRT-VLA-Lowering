"""nn.Module wrappers that call SSM plugin custom ops.

Same idea as ``trt.plugin.attention.PluginAttention``: keep the surrounding
projections in PyTorch, replace the fused kernel body with a custom op.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PluginCausalConv1d(nn.Module):
    """Depthwise causal conv + rolling conv state. Maps to ``causal_conv1d``."""

    def __init__(self, conv_dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv_dim = int(conv_dim)
        self.kernel_size = int(kernel_size)
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=kernel_size,
            groups=conv_dim,
            padding=kernel_size - 1,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trt_edgellm.causal_conv1d.default(
            hidden_states,
            self.conv1d.weight,
            self.conv1d.bias,
            conv_state,
            context_lengths,
            1,
            self.kernel_size - 1,
            1,
            self.conv_dim,
        )


class PluginMambaSSM(nn.Module):
    """Selective scan. Maps to ``update_ssm_state`` (MambaPlugin)."""

    def __init__(self, nheads: int, head_dim: int, dstate: int, ngroups: int, dt_softplus: int = 1):
        super().__init__()
        self.nheads = int(nheads)
        self.head_dim = int(head_dim)
        self.dstate = int(dstate)
        self.ngroups = int(ngroups)
        self.dt_softplus = int(dt_softplus)
        self.A_log = nn.Parameter(torch.randn(nheads))
        self.D = nn.Parameter(torch.ones(nheads))
        self.dt_bias = nn.Parameter(torch.zeros(nheads))

    def forward(
        self,
        x: torch.Tensor,
        ssm_b: torch.Tensor,
        ssm_c: torch.Tensor,
        dt: torch.Tensor,
        state: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ssm_a = -torch.exp(self.A_log.to(torch.float32))
        return torch.ops.trt_edgellm.update_ssm_state.default(
            x,
            ssm_a,
            ssm_b,
            ssm_c,
            self.D.to(dtype=x.dtype),
            dt,
            self.dt_bias.to(dtype=x.dtype),
            state,
            context_lengths,
            self.dt_softplus,
            self.ngroups,
            self.nheads,
            self.head_dim,
            self.dstate,
            0,
        )


class PluginMambaMixer(nn.Module):
    """Nemotron-H-style Mamba mixer: in_proj → causal conv → SSM → out_proj."""

    def __init__(
        self,
        hidden_size: int,
        nheads: int = 4,
        head_dim: int = 64,
        dstate: int = 64,
        ngroups: int = 2,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.nheads = int(nheads)
        self.head_dim = int(head_dim)
        self.dstate = int(dstate)
        self.ngroups = int(ngroups)
        self.d_inner = self.nheads * self.head_dim
        self.d_state_flat = self.ngroups * self.dstate
        self.conv_dim = self.d_inner + 2 * self.d_state_flat
        self.conv = PluginCausalConv1d(self.conv_dim, kernel_size=conv_kernel)
        self.ssm = PluginMambaSSM(nheads, head_dim, dstate, ngroups)
        self.in_proj = nn.Linear(hidden_size, self.d_inner + self.conv_dim + self.nheads, bias=False)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        gate, conv_in, dt = projected.split(
            [self.d_inner, self.conv_dim, self.nheads],
            dim=-1,
        )
        conv_out, conv_state_out = self.conv(conv_in, conv_state, context_lengths)
        conv_out = F.silu(conv_out)
        ssm_x, ssm_b, ssm_c = conv_out.split(
            [self.d_inner, self.d_state_flat, self.d_state_flat],
            dim=-1,
        )
        ssm_out, ssm_state_out = self.ssm(
            ssm_x.view(batch, seq_len, self.nheads, self.head_dim),
            ssm_b.view(batch, seq_len, self.ngroups, self.dstate),
            ssm_c.view(batch, seq_len, self.ngroups, self.dstate),
            dt,
            ssm_state,
            context_lengths,
        )
        ssm_out = ssm_out.reshape(batch, seq_len, self.d_inner)
        hidden = ssm_out * F.silu(gate)
        return self.out_proj(hidden), conv_state_out, ssm_state_out


class PluginGatedDeltaNet(nn.Module):
    """Qwen3.5 GDN linear attention. Maps to ``gated_delta_net``. K=V=128."""

    def __init__(self, num_k_heads: int = 4, num_v_heads: int = 4, head_dim: int = 128):
        super().__init__()
        self.num_k_heads = int(num_k_heads)
        self.num_v_heads = int(num_v_heads)
        self.head_dim = int(head_dim)
        self.A_log = nn.Parameter(torch.randn(num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_v_heads))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        h0: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trt_edgellm.gated_delta_net.default(
            q,
            k,
            v,
            a,
            b,
            self.A_log.to(torch.float32),
            self.dt_bias.to(dtype=v.dtype),
            h0,
            context_lengths,
            self.head_dim,
            self.head_dim,
        )
