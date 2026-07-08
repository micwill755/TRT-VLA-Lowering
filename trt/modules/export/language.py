from __future__ import annotations

import torch
import torch.nn as nn

def _as_tensor(x):
    """Unwrap tuple/list outputs from patched attention modules."""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x

def gather_last_token_hidden(
    hidden_states: torch.Tensor,
    last_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, S, H] at last_token_ids [B] or [B, 1] -> [B, H] for lm_head."""
    if last_token_ids.ndim == 1:
        indices = last_token_ids
    else:
        indices = last_token_ids.squeeze(-1)
    batch_idx = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.long,
    )
    return hidden_states[batch_idx, indices]


class CausalLMExportModule(nn.Module):
    """Edge-LLM causal LM: manual decoder loop -> logits + lm_hidden_states + prefix KV.

    External RoPE and KV-cache controls match ``LLMEngineRunner`` prefill/decode.
    ``select_layer=-1`` (default for GR00T TRT) uses final RMSNorm hidden for
    context; positive values capture an intermediate layer output pre-norm.
    """

    def __init__(
        self,
        lm: nn.Module,
        lm_head: nn.Module,
        *,
        select_layer: int = -1,
    ):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.select_layer = int(select_layer)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        lm_dtype = next(self.lm.parameters()).dtype
        hidden = _as_tensor(inputs_embeds).to(dtype=lm_dtype)
        context_hidden = hidden if self.select_layer == 0 else None
        new_kvs = []

        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                past_key_value=past_key_values[i],
                ctx_len=context_lengths,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = _as_tensor(hidden)
            hidden = residual + hidden

            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = _as_tensor(layer.mlp(hidden))
            hidden = residual + hidden
            new_kvs.append(kv)

            if self.select_layer > 0 and (i + 1) == self.select_layer:
                context_hidden = hidden

        hidden = _as_tensor(self.lm.norm(hidden))
        if context_hidden is None:
            context_hidden = hidden

        last_hidden = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last_hidden).float()

        seq_len = inputs_embeds.shape[1]
        prefix_k = torch.stack(
            [kv[:, 0, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        prefix_v = torch.stack(
            [kv[:, 1, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        return logits, context_hidden, prefix_k, prefix_v

# specific to gr00t, before action another project is required for context embeddings
class ContextProjectionExportModule(nn.Module):
    """eagle_linear -> vlln -> vl_self_attention (matches eager context path)."""

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


class MolmoTextEncoderKVExportModule(nn.Module):
    """Molmo manual decoder loop with MolmoPluginAttention + encoder K/V export."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer
        cfg = transformer.config
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        self.norm_after = bool(getattr(cfg, "norm_after", False))

    def _run_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_after:
            residual = hidden
            hidden, kv = block.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                past_key_value=past_key_value,
                ctx_len=context_lengths,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = _as_tensor(hidden)
            hidden = block.attn_norm(hidden)
            hidden = residual + block.dropout(hidden)

            residual = hidden
            hidden = _as_tensor(block.mlp(hidden))
            hidden = block.ff_norm(hidden)
            hidden = residual + block.dropout(hidden)
            return hidden, kv

        residual = hidden
        hidden = _as_tensor(block.attn_norm(hidden))
        hidden, kv = block.self_attn(
            hidden_states=hidden,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            past_key_value=past_key_value,
            ctx_len=context_lengths,
            kvcache_start_index=kvcache_start_index,
        )
        hidden = _as_tensor(hidden)
        hidden = residual + block.dropout(hidden)

        residual = hidden
        hidden = _as_tensor(block.ff_norm(hidden))
        hidden = _as_tensor(block.mlp(hidden))
        hidden = residual + block.dropout(hidden)
        return hidden, kv

    def forward(
        self,
        inputs_embeds: torch.Tensor,          # [B, S, H], vision already spliced
        rope_rotary_cos_sin: torch.Tensor,    # external RoPE table
        context_lengths: torch.Tensor,        # [B]
        kvcache_start_index: torch.Tensor,    # [] for fresh prefill
        *past_key_values: torch.Tensor,       # one [B,2,KvH,cap,D] per layer
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformer = self.transformer
        hidden = transformer.emb_drop(
            inputs_embeds.to(dtype=next(transformer.parameters()).dtype)
        )
        seq_len = int(hidden.shape[1])
        new_kvs: list[torch.Tensor] = []

        for i, block in enumerate(transformer.blocks):
            hidden, kv = self._run_block(
                block,
                hidden,
                rope_rotary_cos_sin,
                past_key_values[i],
                context_lengths,
                kvcache_start_index,
            )
            new_kvs.append(kv)

        hidden = transformer.ln_f(hidden)

        kv_dim = self.num_kv_heads * self.head_dim
        encoder_k_layers: list[torch.Tensor] = []
        encoder_v_layers: list[torch.Tensor] = []
        for kv in new_kvs:
            encoder_k_layers.append(
                kv[:, 0, :, :seq_len, :]
                .permute(0, 2, 1, 3)
                .reshape(kv.shape[0], seq_len, kv_dim)
                .contiguous()
            )
            encoder_v_layers.append(
                kv[:, 1, :, :seq_len, :]
                .permute(0, 2, 1, 3)
                .reshape(kv.shape[0], seq_len, kv_dim)
                .contiguous()
            )
        encoder_k = torch.stack(encoder_k_layers, dim=0).contiguous()
        encoder_v = torch.stack(encoder_v_layers, dim=0).contiguous()
        return hidden, encoder_k, encoder_v


class MolmoMultimodalEncoderKVExportModule(nn.Module):
    """Vision splice + plugin-backed Molmo text encoder export."""

    def __init__(self, backbone: nn.Module, image_token_positions: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.text_export = MolmoTextEncoderKVExportModule(backbone.transformer)
        self.register_buffer(
            "image_token_positions",
            image_token_positions.to(dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        *kv_caches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

        inputs_embeds = self.backbone.transformer.wte(safe_input_ids)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).clone()
        flat_embeds[self.image_token_positions] = (
            flat_embeds[self.image_token_positions]
            + image_features[: self.image_token_positions.numel()]
        )
        inputs_embeds = flat_embeds.reshape_as(inputs_embeds)

        return self.text_export(
            inputs_embeds,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            *kv_caches,
        )