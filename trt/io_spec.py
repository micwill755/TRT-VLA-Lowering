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
    lm_to_action_slots: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def language_input_names(self, num_kv_caches: int) -> list[str]:
        return list(self.language.input_names) + [
            f"kv_cache_{i}" for i in range(num_kv_caches)
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

    def action_context_input_indices(self) -> tuple[int, ...]:
        return tuple(action_slot for _, action_slot in self.lm_to_action_slots)

    def to_plugin_info(self) -> dict[str, Any]:
        return {
            "vision_input_names": list(self.vision.input_names),
            "vision_output_names": list(self.vision.output_names),
            "language_input_names": list(self.language.input_names),
            "language_output_names": list(self.language.output_names),
            "action_input_names": list(self.action.input_names),
            "action_output_names": list(self.action.output_names),
            "lm_to_action_slots": [list(pair) for pair in self.lm_to_action_slots],
        }


GROOT_EDGE_IO = PipelineIOSpec(
    vision=ComponentIOSpec(
        input_names=("pixel_values",),
        output_names=("visual_embeds",),
    ),
    language=ComponentIOSpec(
        input_names=("inputs_embeds", "ctx_len"),
        output_names=("context_embs",),
    ),
    action=ComponentIOSpec(
        input_names=(
            "actions",
            "timestep",
            "context_embs",
            "state",
            "embodiment_id",
        ),
        output_names=("pred_velocity",),
    ),
    lm_to_action_slots=((0, 2),),
)

PI05_EDGE_IO = PipelineIOSpec(
    vision=ComponentIOSpec(
        input_names=("pixel_values",),
        output_names=("image_embeds",),
    ),
    language=ComponentIOSpec(
        input_names=("inputs_embeds", "ctx_len"),
        output_names=("hidden_states", "prefix_k", "prefix_v"),
    ),
    action=ComponentIOSpec(
        input_names=(
            "x_t",
            "timestep",
            "prefix_k",
            "prefix_v",
            "position_ids",
            "attention_mask",
        ),
        output_names=("velocity",),
    ),
    lm_to_action_slots=((1, 2), (2, 3)),
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


@dataclass(frozen=True)
class LMActionBinding:
    """Maps LLMEngineRunner state to action-engine inputs."""

    source: str  # "kv_cache" | "hidden_states" | "postprocess"
    action_input: str
    dtype: str | None = None
    kv_axis: str | None = None  # e.g. "prefix_k" | "prefix_v" when source is kv_cache


@dataclass(frozen=True)
class VLAPipelineManifest:
    """Runtime wiring manifest for VLA pipelines using LLMEngineRunner."""

    model: str
    lm_runtime: str
    llm_dir: str
    action_dir: str
    vision_dir: str
    lm_action_bindings: tuple[LMActionBinding, ...]
    language_post_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "lm_runtime": self.lm_runtime,
            "llm_dir": self.llm_dir,
            "vision_dir": self.vision_dir,
            "action_dir": self.action_dir,
            "lm_action_bindings": [
                {
                    "source": b.source,
                    "action_input": b.action_input,
                    **({"dtype": b.dtype} if b.dtype else {}),
                    **({"kv_axis": b.kv_axis} if b.kv_axis else {}),
                }
                for b in self.lm_action_bindings
            ],
        }
        if self.language_post_dir is not None:
            payload["language_post_dir"] = self.language_post_dir
        return payload


def pi05_llm_runner_manifest(
    *,
    llm_dir: str = "llm",
    vision_dir: str = "visual",
    action_dir: str = "action",
) -> dict[str, Any]:
    """Manifest for PI0.5: extract prefix K/V from LinearKVCache after LM prefill."""
    manifest = VLAPipelineManifest(
        model="pi05",
        lm_runtime="llm_engine_runner",
        llm_dir=llm_dir,
        vision_dir=vision_dir,
        action_dir=action_dir,
        lm_action_bindings=(
            LMActionBinding(
                source="kv_cache",
                action_input="prefix_k",
                kv_axis="prefix_k",
                dtype="float16",
            ),
            LMActionBinding(
                source="kv_cache",
                action_input="prefix_v",
                kv_axis="prefix_v",
                dtype="float16",
            ),
        ),
    )
    return manifest.to_dict()
