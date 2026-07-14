from __future__ import annotations

import torch
import torch.nn as nn


class Cosmos3MoTBackboneExportModule(nn.Module):
    """MoT backbone: und_seq + gen_seq + rotary -> last_hidden_state.

    Engine 3. No embed or decode heads.
    IN:  und_seq [und_len, hidden], gen_seq [gen_len, hidden], rotary (4 tensors)
    OUT: last_hidden_state [und_len + gen_len, hidden]
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        sample_und_seq: torch.Tensor,
        sample_gen_seq: torch.Tensor,
        sample_rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        super().__init__()
        self.layers = transformer.layers
        self.norm = transformer.norm
        self.norm_moe_gen = transformer.norm_moe_gen

        with torch.no_grad():
            out = self.forward(
                sample_und_seq,
                sample_gen_seq,
                sample_rotary_emb[0],
                sample_rotary_emb[1],
                sample_rotary_emb[2],
                sample_rotary_emb[3],
            )
            self.output_shape = tuple(out.shape)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        cos_und: torch.Tensor,
        sin_und: torch.Tensor,
        cos_gen: torch.Tensor,
        sin_gen: torch.Tensor,
    ) -> torch.Tensor:
        rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)
        return torch.cat([und_out, gen_out], dim=0)
