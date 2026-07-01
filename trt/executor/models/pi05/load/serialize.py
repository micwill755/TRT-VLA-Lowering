from __future__ import annotations


class SerializedPI05Vision:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, pixel_values):
        return self.engine({
            "pixel_values": pixel_values,
        })[0]


class SerializedPI05Language:
    bundles_kv_caches = True

    def __init__(self, engine):
        self.engine = engine
        self.max_seq_len = int(engine.config["max_seq_len"])

    def __call__(
        self,
        input_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        kv_caches,
    ):
        inputs = {
            "inputs_embeds": input_embs,
            "rope_rotary_cos_sin": rope_rotary_cos_sin,
            "context_lengths": ctx_len,
            "kvcache_start_index": kvcache_start_index,
            "last_token_ids": last_token_ids,
        }

        for i, kv_cache in enumerate(kv_caches):
            inputs[f"past_key_values_{i}"] = kv_cache

        outputs = self.engine(inputs)
        return outputs[1], outputs[2], outputs[3]


class SerializedPI05Action:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask):
        return self.engine({
            "x_t": x_t,
            "timestep": timestep,
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        })[0]
