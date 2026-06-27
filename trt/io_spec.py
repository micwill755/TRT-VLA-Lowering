from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ComponentIOSpec:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

@dataclass(frozen=True)
class PipelineIOSpec:
    """Positional tensor IO for vision / language / action TRT engines."""

    vision: ComponentIOSpec
    language: ComponentIOSpec
    action: ComponentIOSpec
    action_context: ComponentIOSpec | None = None
    lm_to_action_slots: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    lm_to_action_context_slots: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    context_to_action_slots: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def language_input_names(self, num_kv_caches: int) -> list[str]:
        return list(self.language.input_names) + [
            f"past_key_values_{i}" for i in range(num_kv_caches)
        ]

    def wire_lm_outputs_to_action(
        self,
        lm_outputs: tuple[Any, ...],
        action_inputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Splice language-engine output tensors into action-engine inputs by slot index."""
        wired = list(action_inputs)
        for lm_slot, action_slot in self.lm_to_action_slots:
            wired[action_slot] = lm_outputs[lm_slot]
        return tuple(wired)

    def wire_lm_outputs_to_action_context(
        self,
        lm_outputs: tuple[Any, ...],
        context_inputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        wired = list(context_inputs)
        for lm_slot, context_slot in self.lm_to_action_context_slots:
            wired[context_slot] = lm_outputs[lm_slot]
        return tuple(wired)

    def wire_context_outputs_to_action(
        self,
        context_outputs: tuple[Any, ...],
        action_inputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        wired = list(action_inputs)
        for context_slot, action_slot in self.context_to_action_slots:
            wired[action_slot] = context_outputs[context_slot]
        return tuple(wired)

    def action_context_input_indices(self) -> tuple[int, ...]:
        if self.lm_to_action_context_slots:
            return tuple(context_slot for _, context_slot in self.lm_to_action_context_slots)
        return tuple(action_slot for _, action_slot in self.lm_to_action_slots)


VLA_VISION_INPUT_NAMES = ("pixel_values",)
VLA_VISION_OUTPUT_NAMES = ("visual_embeds",)
VLA_LANGUAGE_INPUT_NAMES = (
    "inputs_embeds",
    "rope_rotary_cos_sin",
    "context_lengths",
    "kvcache_start_index",
    "last_token_ids",
)
VLA_LANGUAGE_OUTPUT_NAMES = ("logits", "lm_hidden_states", "prefix_k", "prefix_v")
VLA_LANGUAGE_LEADING_INPUT_COUNT = len(VLA_LANGUAGE_INPUT_NAMES)
VLA_ACTION_OUTPUT_NAMES = ("velocity",)

VLA_VISION_IO = ComponentIOSpec(
    input_names=VLA_VISION_INPUT_NAMES,
    output_names=VLA_VISION_OUTPUT_NAMES,
)
VLA_LANGUAGE_IO = ComponentIOSpec(
    input_names=VLA_LANGUAGE_INPUT_NAMES,
    output_names=VLA_LANGUAGE_OUTPUT_NAMES,
)


GROOT_EDGE_IO = PipelineIOSpec(
    vision=VLA_VISION_IO,
    language=VLA_LANGUAGE_IO,
    action_context=ComponentIOSpec(
        input_names=("lm_hidden_states",),
        output_names=("vl_embs",),
    ),
    action=ComponentIOSpec(
        input_names=(
            "actions",
            "timestep",
            "context_embs",
            "state",
            "embodiment_id",
        ),
        output_names=VLA_ACTION_OUTPUT_NAMES,
    ),
    lm_to_action_context_slots=((1, 0),),
    context_to_action_slots=((0, 2),),
)

PI05_EDGE_IO = PipelineIOSpec(
    vision=VLA_VISION_IO,
    language=VLA_LANGUAGE_IO,
    action=ComponentIOSpec(
        input_names=(
            "x_t",
            "timestep",
            "prefix_k",
            "prefix_v",
            "position_ids",
            "attention_mask",
        ),
        output_names=VLA_ACTION_OUTPUT_NAMES,
    ),
    lm_to_action_slots=((1, 2), (2, 3)),
)

MOLMOACT2_BACKBONE_IO = ComponentIOSpec(
    input_names=(
        "input_ids",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "attention_mask",
    ),
    output_names=("encoder_k", "encoder_v"),
)

MOLMOACT2_EDGE_IO = PipelineIOSpec(
    vision=VLA_VISION_IO,
    language=MOLMOACT2_BACKBONE_IO,
    action=ComponentIOSpec(
        input_names=(
            "x_t",
            "timestep",
            "encoder_k",
            "encoder_v",
            "encoder_attention_mask",
        ),
        output_names=VLA_ACTION_OUTPUT_NAMES,
    ),
    lm_to_action_slots=((0, 2), (1, 3)),
)

@dataclass(frozen=True)
class ActionRolloutConfig:
    noise_input_name: str
    timestep_schedule: str
    rollout_dt_sign: int

GROOT_ACTION_ROLLOUT = ActionRolloutConfig(
    noise_input_name="actions",
    timestep_schedule="discrete_buckets",
    rollout_dt_sign=1,
)

PI05_ACTION_ROLLOUT = ActionRolloutConfig(
    noise_input_name="x_t",
    timestep_schedule="continuous_flow",
    rollout_dt_sign=-1,
)

MOLMOACT2_ACTION_ROLLOUT = ActionRolloutConfig(
    noise_input_name="x_t",
    timestep_schedule="continuous_flow",
    rollout_dt_sign=1,
)


def action_rollout_extra_config(
    io: PipelineIOSpec,
    rollout: ActionRolloutConfig,
    *,
    num_steps: int,
    num_timestep_buckets: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "noise_input_name": rollout.noise_input_name,
        "timestep_schedule": rollout.timestep_schedule,
        "rollout_dt_sign": rollout.rollout_dt_sign,
        "lm_to_action_slots": [list(pair) for pair in io.lm_to_action_slots],
    }
    if io.context_to_action_slots:
        config["context_to_action_slots"] = [
            list(pair) for pair in io.context_to_action_slots
        ]
    if io.lm_to_action_context_slots:
        config["lm_to_action_context_slots"] = [
            list(pair) for pair in io.lm_to_action_context_slots
        ]
    if rollout.timestep_schedule == "discrete_buckets":
        config["num_inference_timesteps"] = int(num_steps)
        config["num_timestep_buckets"] = int(num_timestep_buckets or 1)
    else:
        config["num_inference_steps"] = int(num_steps)
    config.update(extra)
    return config


def lm_to_action_slots_from_config(action_config: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    raw = action_config.get("lm_to_action_slots")
    if not raw:
        return ()
    return tuple(tuple(pair) for pair in raw)
