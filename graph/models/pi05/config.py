"""Serialized-engine layout for PI05. Consumed by ``EdgeExporter.save_engines``."""

from exporter import EngineSaveConfig, ModelSaveConfig

VISION = EngineSaveConfig(
    directory="visual",
    engine_file="visual.engine",
    model_type="vit",
    component="vision",
    runtime_op="vision_tower",
    input_names=("pixel_values",),
    output_names=("visual_embeds",),
)

LANGUAGE = EngineSaveConfig(
    directory="language",
    engine_file="language.engine",
    model_type="language",
    component="language",
    runtime_op="text_encoder",
    input_names=(
        "inputs_embeds",
        "rope_rotary_cos_sin",
        "context_lengths",
        "kvcache_start_index",
        "last_token_ids",
        "ds_stack",
    ),
    output_names=("logits", "lm_hidden_states", "prefix_k", "prefix_v"),
)

ACTION = EngineSaveConfig(
    directory="action",
    engine_file="action.engine",
    model_type="action",
    component="diffusion",
    runtime_op="action_expert",
    input_names=(
        "x_t",
        "timestep",
        "prefix_k",
        "prefix_v",
        "position_ids",
        "attention_mask",
    ),
    output_names=("velocity",),
    extra={
        "noise_input_name": "x_t",
        "timestep_schedule": "continuous_flow",
        "rollout_dt_sign": -1,
        "lm_to_action_slots": ((1, 2), (2, 3)),
    },
)

SAVE = ModelSaveConfig(vision=VISION, language=LANGUAGE, action=ACTION)
