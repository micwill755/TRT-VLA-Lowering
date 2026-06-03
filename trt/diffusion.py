import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
from trt.prefix_cache import PrefixKVCache

class GrootStaticDiffusionStep(nn.Module):
    """
    One GR00T denoising step.

    Fuses:
      state_encoder
      action_encoder
      future_tokens
      DiT diffusion model
      action_decoder

    Inputs:
      actions       [B, action_horizon, action_dim]
      timestep      [B]
      vl_embs       [B, vl_seq_len, hidden]
      state         [B, state_horizon, max_state_dim]
      embodiment_id [B]

    Output:
      pred_velocity [B, action_horizon, action_dim]
    """

    def __init__(self, action_head):
        super().__init__()
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
        self.action_decoder = action_head.action_decoder
        self.future_tokens = action_head.future_tokens
        self.model = action_head.model
        self.position_embedding = getattr(action_head, "position_embedding", None)

        self.add_pos_embed = action_head.config.add_pos_embed
        self.action_horizon = action_head.config.action_horizon

    def forward(self, actions, timestep, vl_embs, state, embodiment_id):
        state_features = self.state_encoder(state, embodiment_id)
        action_features = self.action_encoder(actions, timestep, embodiment_id)

        if self.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device,
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0],
            -1,
            -1,
        )

        sa_embs = torch.cat(
            (state_features, future_tokens, action_features),
            dim=1,
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timestep,
        )

        pred = self.action_decoder(model_output, embodiment_id)
        return pred[:, -self.action_horizon :]

class PI05StaticKVDiffusionStep(nn.Module):
    def __init__(self, pi05):
        super().__init__()
        self.config = pi05.config
        self.action_in_proj = pi05.action_in_proj
        self.time_mlp_in = pi05.time_mlp_in
        self.time_mlp_out = pi05.time_mlp_out
        self.expert = pi05.paligemma_with_expert.gemma_expert.model
        self.action_out_proj = pi05.action_out_proj
        self.chunk_size = pi05.config.chunk_size
        self.hidden_size = pi05.action_in_proj.out_features

    def embed_suffix(self, x_t, timestep):
        action_emb = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=action_emb.dtype)

        adarms_cond = self.time_mlp_in(time_emb)
        adarms_cond = F.silu(adarms_cond)
        adarms_cond = self.time_mlp_out(adarms_cond)
        adarms_cond = F.silu(adarms_cond)

        return action_emb, adarms_cond

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        suffix_embs, adarms_cond = self.embed_suffix(x_t, timestep)
        past_key_values = PrefixKVCache(prefix_k, prefix_v)

        out = self.expert(
            inputs_embeds=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = out.last_hidden_state[:, -self.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

class SmolVLAStaticKVDiffusionStep(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.config = core.config
        self.vlm_with_expert = core.vlm_with_expert
        self.action_in_proj = core.action_in_proj
        self.action_time_mlp_in = core.action_time_mlp_in
        self.action_time_mlp_out = core.action_time_mlp_out
        self.action_out_proj = core.action_out_proj
        self.chunk_size = core.config.chunk_size
        self.hidden_size = core.vlm_with_expert.expert_hidden_size
        self.num_layers = core.vlm_with_expert.num_vlm_layers

    def embed_suffix(self, x_t, timestep):
        action_emb = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=timestep.device,
        ).to(dtype=action_emb.dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        return action_time_emb

    def forward(self, x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask):
        suffix_embs = self.embed_suffix(x_t, timestep)

        past_key_values = {
            i: {
                "key_states": prefix_k[i],
                "value_states": prefix_v[i],
            }
            for i in range(self.num_layers)
        }

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )

        suffix_out = outputs_embeds[1][:, -self.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)