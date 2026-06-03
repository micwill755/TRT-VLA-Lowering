import torch
import torch.nn as nn


class PI05PrefixLanguagePrefill(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.language_model = core.paligemma_with_expert.paligemma.model.language_model

    def forward(self, prefix_embs, attention_mask, position_ids):
        out = self.language_model(
            inputs_embeds=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
        )
        cache = out.past_key_values
        prefix_k = torch.stack([layer.keys for layer in cache.layers], dim=0)
        prefix_v = torch.stack([layer.values for layer in cache.layers], dim=0)
        return prefix_k, prefix_v


class GROOTLanguageEmbed(nn.Module):
    def __init__(self, groot):
        super().__init__()
        self.eagle_model = groot.backbone.eagle_model
        self.eagle_linear = groot.backbone.eagle_linear
        self.select_layer = groot.backbone.select_layer
        self.vlln = groot.action_head.vlln
        self.vl_self_attention = groot.action_head.vl_self_attention
        self.image_token_index = self.eagle_model.image_token_index

    def forward(self, input_ids, attention_mask, vit_embeds):
        input_embeds = self.eagle_model.language_model.get_input_embeddings()(input_ids)

        image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
        image_mask = image_mask.expand_as(input_embeds)

        input_embeds = input_embeds.masked_scatter(
            image_mask,
            vit_embeds.reshape(-1).to(device=input_embeds.device, dtype=input_embeds.dtype),
        )

        out = self.eagle_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        vl_embs = out.hidden_states[self.select_layer]
        vl_embs = self.eagle_linear(vl_embs)
        vl_embs = self.vlln(vl_embs)
        vl_embs = self.vl_self_attention(vl_embs)
        return vl_embs

class SmolVLAPrefixLanguagePrefill(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert
        self.num_layers = core.vlm_with_expert.num_vlm_layers

    def forward(self, prefix_embs, attention_mask, position_ids):
        _, cache = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        prefix_k = torch.stack([cache[i]["key_states"] for i in range(self.num_layers)], dim=0)
        prefix_v = torch.stack([cache[i]["value_states"] for i in range(self.num_layers)], dim=0)
        return prefix_k, prefix_v