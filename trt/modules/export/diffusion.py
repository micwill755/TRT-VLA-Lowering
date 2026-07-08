from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
from trt.prefix_cache import PrefixKVCache, SmolVLAPrefixPastLayers


class TRTFixedCategorySpecificLinearExportModule(nn.Module):
    """Freeze one GR00T embodiment-specific Linear into a normal Linear.

    GR00T stores one weight matrix per robot embodiment and selects it with
    embodiment_id at runtime. For TensorRT deployment we compile one robot at a
    time, so this wrapper picks that robot's weights once in __init__ and the
    forward path becomes a plain static F.linear.
    """

    def __init__(self, layer: nn.Module, embodiment_id: torch.Tensor):
        super().__init__()

        cat_id = int(embodiment_id.flatten()[0].item())

        # Original: [num_embodiments, input_dim, output_dim]
        # using cat_id selects the weight matrix for one embodiment/robot -> [input_dim, output_dim]
        # nn.functional.linear expects -> weight: [output_dim, input_dim] so we transpose
        weight = layer.W[cat_id].transpose(0, 1).contiguous()
        bias = layer.b[cat_id].contiguous()

        # detach() breaks any autograd link to the original multi-embodiment
        # parameter; clone() gives this fixed wrapper independent storage for
        # the selected slice. This copy happens once during wrapper creation,
        # not in forward, and lets TensorRT see normal immutable weights.
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.out_features = int(bias.shape[0])

    def forward(self, x):
        # x: [B, T, input_dim]
        # out: [B, T, output_dim]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size * seq_len, x.shape[-1])
        x = F.linear(x, self.weight, self.bias)
        return x.reshape(batch_size, seq_len, self.out_features)

class TRTDynamicCategorySpecificLinearExportModule(nn.Module):
    """TensorRT-friendly dynamic version of GR00T CategorySpecificLinear.

    Unlike the fixed wrapper, this keeps the full embodiment weight bank and
    uses runtime embodiment_id values to gather W/b for each batch item. The
    math stays equivalent to GR00T category-specific linear:
    x [B,T,in] @ W[embodiment] [B,in,out] + b[embodiment].
    """

    def __init__(self, layer: nn.Module):
        super().__init__()

        # Keep the full embodiment weight bank.
        # W: [num_embodiments, input_dim, output_dim]
        # b: [num_embodiments, output_dim]
        self.W = layer.W
        self.b = layer.b

    def forward(self, x, cat_ids):
        # x:       [B, T, input_dim]
        # cat_ids: [B]

        cat_ids = cat_ids.to(dtype=torch.long)

        # selected_w: [B, input_dim, output_dim]
        # selected_b: [B, output_dim]
        selected_w = torch.index_select(self.W, dim=0, index=cat_ids)
        selected_b = torch.index_select(self.b, dim=0, index=cat_ids)

        # out: [B, T, output_dim]
        out = torch.bmm(x, selected_w)

        # bias: [B, 1, output_dim], broadcast over T
        return out + selected_b.unsqueeze(1)


class TRTDynamicCategorySpecificMLPExportModule(nn.Module):
    """Dynamic two-layer category-specific MLP used by GR00T.

    GR00T state encoders and action decoders are CategorySpecificMLP modules:
    each contains layer1/layer2 CategorySpecificLinear layers. This wrapper
    preserves runtime embodiment selection for both layers.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.layer1 = TRTDynamicCategorySpecificLinearExportModule(mlp.layer1)
        self.layer2 = TRTDynamicCategorySpecificLinearExportModule(mlp.layer2)

    def forward(self, x, embodiment_id):
        hidden = F.relu(self.layer1(x, embodiment_id))
        return self.layer2(hidden, embodiment_id)

class TRTGrootActionEncoderExportModule(nn.Module):
    def __init__(self, action_encoder: nn.Module, embodiment_id: torch.Tensor):
        super().__init__()
        self.W1 = TRTFixedCategorySpecificLinearExportModule(action_encoder.W1, embodiment_id)
        self.W2 = TRTFixedCategorySpecificLinearExportModule(action_encoder.W2, embodiment_id)
        self.W3 = TRTFixedCategorySpecificLinearExportModule(action_encoder.W3, embodiment_id)
        self.pos_encoding = action_encoder.pos_encoding

    def forward(self, actions, timesteps, embodiment_id):
        batch_size, action_horizon, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == batch_size:
            timesteps = timesteps.unsqueeze(1).expand(-1, action_horizon)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,).")

        action_emb = self.W1(actions)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        hidden = torch.cat([action_emb, timestep_emb], dim=-1)
        hidden = F.silu(self.W2(hidden))
        return self.W3(hidden)

class TRTDynamicGrootActionEncoderExportModule(nn.Module):
    """Dynamic GR00T noisy-action encoder.

    The original action encoder uses three embodiment-specific linear layers
    around the action embedding, timestep positional embedding, and SiLU block.
    This wrapper keeps embodiment_id dynamic while spelling the category-specific
    pieces as index_select + bmm so Torch-TRT can lower them reliably.
    """

    def __init__(self, action_encoder: nn.Module):
        super().__init__()
        self.W1 = TRTDynamicCategorySpecificLinearExportModule(action_encoder.W1)
        self.W2 = TRTDynamicCategorySpecificLinearExportModule(action_encoder.W2)
        self.W3 = TRTDynamicCategorySpecificLinearExportModule(action_encoder.W3)
        self.pos_encoding = action_encoder.pos_encoding

    def forward(self, actions, timesteps, embodiment_id):
        batch_size, action_horizon, _ = actions.shape

        timesteps = timesteps.unsqueeze(1).expand(-1, action_horizon)

        action_emb = self.W1(actions, embodiment_id)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)

        hidden = torch.cat([action_emb, timestep_emb], dim=-1)
        hidden = F.silu(self.W2(hidden, embodiment_id))
        return self.W3(hidden, embodiment_id)

class ActionStepEncoderExportModule(nn.Module):
    """Base contract for model-specific action-step encoding.

    Subclasses implement forward() to turn noisy actions, timestep, and
    model-specific context tensors into the args/kwargs consumed by the action
    expert and velocity decoder. The default helpers cover common expert output
    and velocity shapes, while model-specific encoders can override them.
    """

    def get_action_hidden(self, expert_out, output_tokens: int):
        # Default path for experts that return either a standard model output
        # with last_hidden_state, a raw hidden-state tensor, or a tuple/list whose
        # first item is the hidden-state tensor.
        hidden = (
            expert_out.last_hidden_state
            if hasattr(expert_out, "last_hidden_state")
            else expert_out
        )

        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]

        return hidden[:, -output_tokens:]

    def process_velocity(self, velocity):
        # Default path for models whose decoder already returns the final action
        # velocity shape. Override for models that need reshaping or cropping.
        return velocity

class StaticActionVelocityStepExportModule(nn.Module):
    """One static denoising step shared by VLA action diffusion modules.

    The model-specific step_encoder owns the messy part: converting noisy
    actions, timestep, and context tensors into the exact action_expert call.
    This wrapper only runs the expert, selects action-token hidden states, and
    decodes those hidden states into a velocity update.
    """

    def __init__(
        self,
        *,
        step_encoder: ActionStepEncoderExportModule,
        action_expert: nn.Module,
        velocity_decoder: nn.Module,
        output_tokens: int,
        cast_hidden_fp32: bool = True,
    ):
        super().__init__()
        self.step_encoder = step_encoder
        self.action_expert = action_expert
        self.velocity_decoder = velocity_decoder
        self.output_tokens = int(output_tokens)
        self.cast_hidden_fp32 = cast_hidden_fp32

    def forward(self, x_t, timestep, *inputs):
        # Build the action expert inputs and any decoder-specific side inputs.
        expert_args, expert_kwargs, decoder_args, decoder_kwargs = self.step_encoder(
            x_t,
            timestep,
            *inputs,
        )

        # Run the model-specific action expert: Gemma expert, DiT, etc.
        expert_out = self.action_expert(*expert_args, **expert_kwargs)

        # Most experts return last_hidden_state, but some wrappers return tuples
        # or need custom suffix-token selection.
        action_hidden = self.step_encoder.get_action_hidden(
            expert_out,
            self.output_tokens,
        )

        if self.cast_hidden_fp32:
            action_hidden = action_hidden.to(dtype=torch.float32)

        # Project action-token hidden states back to action-space velocity.
        velocity = self.velocity_decoder(
            action_hidden,
            *decoder_args,
            **decoder_kwargs,
        )

        return self.step_encoder.process_velocity(velocity)


class GrootDiTStepEncoderExportModule(ActionStepEncoderExportModule):
    def __init__(self, action_head, embodiment_id: torch.Tensor | None = None):
        super().__init__()
        if embodiment_id is None:
            self.state_encoder = action_head.state_encoder
            self.action_encoder = action_head.action_encoder
        else:
            # Keep embodiment_id as a runtime input while replacing GR00T's
            # category-specific modules with Torch-TRT-friendly dynamic wrappers.
            self.state_encoder = TRTDynamicCategorySpecificMLPExportModule(
                action_head.state_encoder
            )
            self.action_encoder = TRTDynamicGrootActionEncoderExportModule(
                action_head.action_encoder
            )
        self.future_tokens = action_head.future_tokens
        self.position_embedding = getattr(action_head, "position_embedding", None)
        self.add_pos_embed = action_head.config.add_pos_embed

    def forward(self, actions, timestep, vl_embs, state, embodiment_id):
        state_features = self.state_encoder(state, embodiment_id)
        action_features = self.action_encoder(actions, timestep, embodiment_id)

        if self.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device,
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0],
            -1,
            -1,
        )

        sa_embs = torch.cat(
            (state_features, future_tokens, action_features),
            dim=1,
        )

        expert_args = ()
        expert_kwargs = {
            "hidden_states": sa_embs,
            "encoder_hidden_states": vl_embs,
            "timestep": timestep,
        }

        decoder_args = (embodiment_id,)
        decoder_kwargs = {}

        return expert_args, expert_kwargs, decoder_args, decoder_kwargs


class PI05PrefixKVStepEncoderExportModule(ActionStepEncoderExportModule):
    """PI05 suffix embed + AdaRMS cond, consumed by Gemma action expert."""

    def __init__(self, core):
        super().__init__()
        self.action_in_proj = core.action_in_proj
        self.time_mlp_in = core.time_mlp_in
        self.time_mlp_out = core.time_mlp_out
        self.config = core.config
        self.hidden_size = core.action_in_proj.out_features

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        suffix_embs = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=suffix_embs.dtype)

        adarms_cond = self.time_mlp_in(time_emb)
        adarms_cond = F.silu(adarms_cond)
        adarms_cond = self.time_mlp_out(adarms_cond)
        adarms_cond = F.silu(adarms_cond)

        expert_kwargs = {
            "inputs_embeds": suffix_embs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": PrefixKVCache(prefix_k, prefix_v),
            "use_cache": False,
            "adarms_cond": adarms_cond,
        }
        return (), expert_kwargs, (), {}


class AlpamayoPrefixKVStepEncoderExportModule(ActionStepEncoderExportModule):
    """Alpamayo suffix embed + prefix KV, consumed by the Qwen3-VL expert."""

    def __init__(self, model):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.action_space_dims = tuple(int(x) for x in model.action_space.get_action_space_dims())
        self.n_diffusion_tokens = int(self.action_space_dims[0])

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        batch_size = x_t.shape[0]
        if timestep.ndim == 1:
            t = timestep.view(batch_size, 1, 1).to(dtype=x_t.dtype, device=x_t.device)
        else:
            t = timestep.to(dtype=x_t.dtype, device=x_t.device)

        suffix_embs = self.action_in_proj(x_t, t)
        if suffix_embs.dim() == 2:
            suffix_embs = suffix_embs.view(batch_size, self.n_diffusion_tokens, -1)

        expert_kwargs = {
            "inputs_embeds": suffix_embs,
            "position_ids": position_ids,
            "past_key_values": PrefixKVCache(prefix_k, prefix_v),
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        return (), expert_kwargs, (), {}

    def process_velocity(self, velocity):
        return velocity.view(-1, *self.action_space_dims)


class SmolVLAPrefixKVStepEncoderExportModule(ActionStepEncoderExportModule):
    """SmolVLA suffix embed + prefix KV, consumed by ``vlm_with_expert`` expert path."""

    def __init__(self, core):
        super().__init__()
        self.action_in_proj = core.action_in_proj
        self.action_time_mlp_in = core.action_time_mlp_in
        self.action_time_mlp_out = core.action_time_mlp_out
        self.config = core.config
        self.expert_hidden_size = core.vlm_with_expert.expert_hidden_size

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        suffix_embs = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=suffix_embs.dtype)
        time_emb = time_emb[:, None, :].expand_as(suffix_embs)
        action_time_emb = torch.cat([suffix_embs, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        suffix_embs = self.action_time_mlp_out(action_time_emb)

        expert_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": SmolVLAPrefixPastLayers(prefix_k, prefix_v),
            "inputs_embeds": [None, suffix_embs],
            "use_cache": True,
            "fill_kv_cache": False,
        }
        return (), expert_kwargs, (), {}


class SmolVLAExpertExportModule(nn.Module):
    """Suffix-only ``vlm_with_expert.forward`` for one denoising step."""

    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        self.vlm_with_expert = vlm_with_expert

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        inputs_embeds,
        use_cache: bool = True,
        fill_kv_cache: bool = False,
    ):
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        return outputs_embeds[1]


def _molmo_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_action_expert_rope_export(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE when ``cos`` / ``sin`` are already ``[B, H, S, D//2]``."""
    half = int(cos.shape[-1])
    q1, q2 = torch.split(q, half, dim=-1)
    k1, k2 = torch.split(k, half, dim=-1)
    q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q, k


class ActionExpertSelfAttentionExportModule(nn.Module):
    """Export wrapper for Molmo ActionExpert self-attention (qkv + RoPE + SDPA)."""

    def __init__(self, self_attn: nn.Module, *, is_causal: bool = False):
        super().__init__()
        self.qkv = self_attn.qkv
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm
        self.out_proj = self_attn.out_proj
        self.num_heads = int(self_attn.num_heads)
        self.head_dim = int(self_attn.head_dim)
        self.hidden_size = int(self_attn.hidden_size)
        self.is_causal = bool(is_causal)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].contiguous()

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos.numel() > 0:
            q, k = apply_action_expert_rope_export(q, k, rope_cos, rope_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        mask = attn_mask if attn_mask is not None and attn_mask.numel() > 0 else None
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.out_proj(out)


class ActionExpertBlockExportModule(nn.Module):
    """One ActionExpert block with TRT-friendly self-attention."""

    def __init__(self, block: nn.Module, *, is_causal: bool = False):
        super().__init__()
        self.self_norm = block.self_norm
        self.cross_norm = block.cross_norm
        self.ff_norm = block.ff_norm
        self.self_attn = ActionExpertSelfAttentionExportModule(
            block.self_attn,
            is_causal=is_causal,
        )
        self.cross_attn = block.cross_attn
        self.mlp = block.mlp

    def forward(
        self,
        x: torch.Tensor,
        block_modulation: tuple[torch.Tensor, ...],
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = block_modulation

        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            _molmo_modulate(self.self_norm(x), shift_msa, scale_msa),
            self_attn_mask if self_attn_mask is not None and self_attn_mask.numel() > 0 else None,
            rope_cos,
            rope_sin,
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            _molmo_modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0],
            kv_v=cross_kv[1],
            attn_mask=cross_attn_mask if cross_attn_mask is not None and cross_attn_mask.numel() > 0 else None,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _molmo_modulate(self.ff_norm(x), shift_mlp, scale_mlp)
        )
        return x


def build_molmo_action_modulation(
    action_expert: nn.Module,
    timestep: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bake per-block and final AdaLN modulation tensors for one flow step."""
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)
    timestep = timestep.to(dtype=torch.float32)
    conditioning = action_expert._time_conditioning(timestep)
    if dtype is not None:
        conditioning = conditioning.to(dtype=dtype)

    block_mods: list[torch.Tensor] = []
    for block in action_expert.blocks:
        chunks = block.modulation(conditioning).chunk(9, dim=1)
        block_mods.append(torch.stack(chunks, dim=0))
    block_mod = torch.stack(block_mods, dim=0).contiguous()

    final_shift, final_scale = action_expert.final_layer.modulation(conditioning).chunk(2, dim=1)
    final_mod = torch.stack((final_shift, final_scale), dim=0).contiguous()
    return block_mod, final_mod


def build_molmo_action_context(
    action_expert: nn.Module,
    encoder_k: torch.Tensor,
    encoder_v: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
    action_horizon: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project stacked encoder K/V into action-expert cross-attn tensors."""
    num_layers = int(encoder_k.shape[0])
    encoder_kv_states = [(encoder_k[i], encoder_v[i]) for i in range(num_layers)]
    batch_size = int(encoder_k.shape[1])

    kv_contexts = action_expert._prepare_kv_context(encoder_kv_states)
    context_k = torch.stack([k for k, _ in kv_contexts], dim=0).contiguous()
    context_v = torch.stack([v for _, v in kv_contexts], dim=0).contiguous()

    cross_mask = action_expert._build_cross_attention_mask(
        encoder_attention_mask,
        batch_size,
        dtype,
    )
    if (
        encoder_attention_mask is not None
        and bool(encoder_attention_mask.to(dtype=torch.bool).all().item())
    ):
        cross_mask = None

    self_mask = action_expert._build_self_attention_mask(
        None,
        int(action_horizon),
        device,
        dtype,
    )

    rope_cache = None
    if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
        rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
            seq_len=int(action_horizon),
            device=device,
            dtype=dtype,
        )

    if rope_cache is None:
        rope_cos = torch.empty(0, device=device, dtype=dtype)
        rope_sin = torch.empty(0, device=device, dtype=dtype)
    else:
        cos, sin = rope_cache
        half = int(cos.shape[-1])
        num_heads = int(action_expert.blocks[0].self_attn.num_heads)
        rope_cos = (
            cos.reshape(1, 1, int(action_horizon), half)
            .expand(batch_size, num_heads, int(action_horizon), half)
            .contiguous()
        )
        rope_sin = (
            sin.reshape(1, 1, int(action_horizon), half)
            .expand(batch_size, num_heads, int(action_horizon), half)
            .contiguous()
        )

    if cross_mask is None:
        cross_mask = torch.empty(0, device=device, dtype=dtype)
    if self_mask is None:
        self_mask = torch.empty(0, device=device, dtype=dtype)

    return context_k, context_v, cross_mask, self_mask, rope_cos, rope_sin


class MolmoAct2ActionFlowStepExportModule(nn.Module):
    """One MolmoAct2 flow-matching velocity step with pre-baked context tensors."""

    def __init__(self, action_expert: nn.Module):
        super().__init__()
        self.action_expert = action_expert
        self.block_exports = nn.ModuleList(
            ActionExpertBlockExportModule(block, is_causal=bool(action_expert.config.causal_attn))
            for block in action_expert.blocks
        )

    def forward(
        self,
        actions: torch.Tensor,
        block_mod: torch.Tensor,
        final_mod: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        cross_mask: torch.Tensor,
        self_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        expert = self.action_expert
        x = expert.action_embed(actions)

        cross_attn_mask = cross_mask if cross_mask.numel() > 0 else None
        self_attn_mask = self_mask if self_mask.numel() > 0 else None

        for idx, block in enumerate(self.block_exports):
            block_modulation = (
                block_mod[idx, 0],
                block_mod[idx, 1],
                block_mod[idx, 2],
                block_mod[idx, 3],
                block_mod[idx, 4],
                block_mod[idx, 5],
                block_mod[idx, 6],
                block_mod[idx, 7],
                block_mod[idx, 8],
            )
            x = block(
                x,
                block_modulation,
                cross_kv=(context_k[idx], context_v[idx]),
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        final_modulation = (final_mod[0], final_mod[1])
        return expert.final_layer(x, final_mod[0], modulation=final_modulation)
