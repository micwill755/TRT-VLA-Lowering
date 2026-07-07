from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Edge-LLM diffusion engine export
# ---------------------------------------------------------------------------

DEFAULT_DIFFUSION_TRT_SETTINGS: dict[str, Any] = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "offload_module_to_cpu": True,
    "use_fp32_acc": True,
}

# Shorter aliases for parity harnesses and docs.
StaticActionVelocityStep = StaticActionVelocityStepExportModule
GrootDiTStepEncoder = GrootDiTStepEncoderExportModule
TRTDynamicCategorySpecificMLP = TRTDynamicCategorySpecificMLPExportModule
