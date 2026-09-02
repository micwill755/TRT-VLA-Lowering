"""Serialized-engine layout for GR00T. Consumed by ``EdgeExporter.save_engines``."""

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

ACTION_CONTEXT = EngineSaveConfig(
    directory="action_context",
    engine_file="action_context.engine",
    model_type="action_context",
    component="action_context",
    runtime_op="action_context",
    input_names=("lm_hidden_states",),
    output_names=("vl_embs",),
)

ACTION = EngineSaveConfig(
    directory="action",
    engine_file="action.engine",
    model_type="action",
    component="diffusion",
    runtime_op="action_expert",
    input_names=(
        "actions",
        "timestep",
        "context_embs",
        "state",
        "embodiment_id",
    ),
    output_names=("velocity",),
    extra={
        "engine_role": "single_action_denoising_step",
        "noise_input_name": "actions",
        "timestep_schedule": "discrete_buckets",
        "rollout_dt_sign": 1,
    },
)

SAVE = ModelSaveConfig(
    vision=VISION,
    language=LANGUAGE,
    action=ACTION,
    action_context=ACTION_CONTEXT,
)
