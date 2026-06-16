import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.pi05.modeling_pi05 import get_safe_dtype
from trt.prefix_cache import PrefixKVCache

def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

class TRTFixedCategorySpecificLinear(nn.Module):
    def __init__(self, layer: nn.Module, embodiment_id: torch.Tensor):
        super().__init__()

        cat_id = int(embodiment_id.flatten()[0].item())

        # Original:
        #   layer.W[cat_id]: [input_dim, output_dim]
        # nn.functional.linear expects:
        #   weight: [output_dim, input_dim]
        weight = layer.W[cat_id].transpose(0, 1).contiguous()
        bias = layer.b[cat_id].contiguous()

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

class TRTGrootActionDecoder(nn.Module):
    def __init__(self, action_decoder: nn.Module, embodiment_id: torch.Tensor):
        super().__init__()
        self.layer1 = TRTFixedCategorySpecificLinear(
            action_decoder.layer1,
            embodiment_id,
        )
        self.layer2 = TRTFixedCategorySpecificLinear(
            action_decoder.layer2,
            embodiment_id,
        )

    def forward(self, x, embodiment_id):
        # Keep embodiment_id in the signature so StaticActionVelocityStep does
        # not need to change. It is ignored because weights are already fixed.
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)

class ActionStepEncoder(nn.Module):
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

class StaticActionVelocityStep(nn.Module):
    """One static denoising step shared by VLA action diffusion modules.

    The model-specific step_encoder owns the messy part: converting noisy
    actions, timestep, and context tensors into the exact action_expert call.
    This wrapper only runs the expert, selects action-token hidden states, and
    decodes those hidden states into a velocity update.
    """

    def __init__(
        self,
        *,
        step_encoder: ActionStepEncoder,
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

class PrefixKVStepEncoder(ActionStepEncoder):
    def __init__(self, action_embedder: nn.Module):
        super().__init__()
        self.action_embedder = action_embedder

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        suffix_embs = self.action_embedder(x_t, timestep)

        expert_args = ()
        expert_kwargs = {
            "inputs_embeds": suffix_embs,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": PrefixKVCache(prefix_k, prefix_v),
            "use_cache": False,
        }

        decoder_args = ()
        decoder_kwargs = {}

        return expert_args, expert_kwargs, decoder_args, decoder_kwargs

class PI05PrefixKVStepEncoder(ActionStepEncoder):
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

class SmolVLAPrefixKVStepEncoder(ActionStepEncoder):
    def __init__(self, core):
        super().__init__()
        self.action_in_proj = core.action_in_proj
        self.action_time_mlp_in = core.action_time_mlp_in
        self.action_time_mlp_out = core.action_time_mlp_out
        self.config = core.config
        self.hidden_size = core.vlm_with_expert.expert_hidden_size

    def forward(self, x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask):
        action_emb = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=action_emb.dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        suffix_embs = torch.cat([action_emb, time_emb], dim=-1)
        suffix_embs = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(suffix_embs)))

        expert_kwargs = {
            "inputs_embeds": [None, suffix_embs],
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": PrefixKVCache(prefix_k, prefix_v),
            "use_cache": False,
            "fill_kv_cache": False,
        }
        return (), expert_kwargs, (), {}

    def get_action_hidden(self, expert_out, output_tokens: int):
        outputs_embeds, _ = expert_out
        suffix_out = outputs_embeds[1]
        return suffix_out[:, -output_tokens:]

class AlpamayoPrefixKVStepEncoder(ActionStepEncoder):
    def __init__(self, model):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.action_space_dims = model.action_space.get_action_space_dims()
        self.n_diffusion_tokens = self.action_space_dims[0]

    def forward(self, x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask):
        suffix_embs = self.action_in_proj(x_t, timestep)

        if suffix_embs.dim() == 2:
            suffix_embs = suffix_embs.view(x_t.shape[0], self.n_diffusion_tokens, -1)

        expert_kwargs = {
            "inputs_embeds": suffix_embs,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": PrefixKVCache(prefix_k, prefix_v),
            "use_cache": False,
        }
        return (), expert_kwargs, (), {}

    def process_velocity(self, velocity):
        return velocity.view(-1, *self.action_space_dims)

class GrootDiTStepEncoder(ActionStepEncoder):
    def __init__(self, action_head):
        super().__init__()
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
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
