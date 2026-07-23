from __future__ import annotations

from trt.serialize import SerializedPositionalEngine

class SerializedPi05Vision:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, pixel_values):
        from trt.vision import VIT_ENGINE_INPUT_NAME, is_nchw_pixel_values, nchw_to_hwc

        images = nchw_to_hwc(pixel_values) if is_nchw_pixel_values(pixel_values) else pixel_values
        input_name = VIT_ENGINE_INPUT_NAME
        if input_name not in self.engine.config_input_names:
            input_name = self.engine.config_input_names[0]
        return self.engine({input_name: images})[0]

class SerializedPi05Language:
    bundles_kv_caches = True

    def __init__(self, engine):
        self.engine = engine
        self.max_seq_len = int(engine.config["max_seq_len"])

    def __call__(self, *args):
        if len(args) < 7:
            raise ValueError(
                f"SerializedPi05Language expected at least 7 inputs, got {len(args)}"
            )
        (
            input_embs,
            rope_rotary_cos_sin,
            ctx_len,
            kvcache_start_index,
            last_token_ids,
            ds_stack,
            *kv_caches,
        ) = args
        inputs = {
            "inputs_embeds": input_embs,
            "rope_rotary_cos_sin": rope_rotary_cos_sin,
            "context_lengths": ctx_len,
            "kvcache_start_index": kvcache_start_index,
            "last_token_ids": last_token_ids,
            "ds_stack": ds_stack,
        }

        for i, kv_cache in enumerate(kv_caches):
            inputs[f"past_key_values_{i}"] = kv_cache

        outputs = self.engine(inputs)
        if len(outputs) < 4:
            raise ValueError(
                f"SerializedPi05Language expected 4 outputs, got {len(outputs)}"
            )
        return outputs[0], outputs[1], outputs[2], outputs[3]

class SerializedPi05Action(SerializedPositionalEngine):
    def __call__(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        return super().__call__(
            x_t,
            timestep,
            prefix_k,
            prefix_v,
            position_ids,
            attention_mask,
        )[0]