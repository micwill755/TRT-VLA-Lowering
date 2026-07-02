from __future__ import annotations

from trt.serialize import SerializedPositionalEngine


class SerializedGrootVision:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, pixel_values):
        from trt.vision import VIT_ENGINE_INPUT_NAME, is_nchw_pixel_values, nchw_to_hwc

        images = nchw_to_hwc(pixel_values) if is_nchw_pixel_values(pixel_values) else pixel_values
        input_name = VIT_ENGINE_INPUT_NAME
        if input_name not in self.engine.config_input_names:
            input_name = self.engine.config_input_names[0]
        return self.engine({input_name: images})[0]


class SerializedGrootLanguage:
    bundles_kv_caches = True

    def __init__(self, engine):
        self.engine = engine
        self.max_seq_len = int(engine.config["max_seq_len"])

    def __call__(self, *args):
        if len(args) < 6:
            raise ValueError(
                f"SerializedGrootLanguage expected at least 6 inputs, got {len(args)}"
            )
        (
            input_embs,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            last_token_ids,
            *kv_caches,
        ) = args
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
        if len(outputs) == 1:
            return outputs[0]
        return outputs[0], outputs[1]

    @property
    def context_output_index(self) -> int:
        output_names = self.engine.config.get("output_names", [])
        for name in ("context_embs", "lm_hidden_states", "vl_embs"):
            if name in output_names:
                return output_names.index(name)
        return 1

    @property
    def lm_hidden_output_index(self) -> int:
        return self.context_output_index


class SerializedGrootActionContext(SerializedPositionalEngine):
    def __call__(self, lm_hidden_states):
        return super().__call__(lm_hidden_states)[0]


class SerializedGrootAction(SerializedPositionalEngine):
    def __call__(self, actions, timestep, context_embs, state, embodiment_id):
        return super().__call__(
            actions,
            timestep,
            context_embs,
            state,
            embodiment_id,
        )[0]
