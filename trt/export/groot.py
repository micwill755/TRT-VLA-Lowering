"""GR00T-specific export hooks and component builders."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import torch
import torch.nn as nn

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.compile import dump_edge_fixture as write_edge_fixture
from trt.data import pack_state
from trt.diffusion_builders import (
    build_groot_diffusion_export_params,
    make_groot_static_action_module,
)
make_static_action_module = make_groot_static_action_module
from trt.export.context import ComponentBuild, ExportContext
from trt.export.hooks import VLAExportHooks
from trt.export.mode import ExportMode
from trt.export.settings import ACTION_TRT_SETTINGS, VISION_TRT_SETTINGS, in_memory_settings
from trt.export.sinks import ExportSink
from trt.io_spec import GROOT_EDGE_IO, PipelineIOSpec
from trt.language import (
    compile_language_trt_with_plugin,
    language_head_dim,
    make_action_context_module,
    make_language_context_wrapper,
)
from trt.language_builders import build_groot_language_export_params
from trt.measure import compute_action_parity_metrics, tensor_error_metrics
from trt.packing import pack_groot_language_inputs
from trt.rope import make_rope_rotary_cos_sin
from trt.serialize import (
    SerializedGrootAction,
    SerializedGrootActionContext,
    SerializedGrootLanguage,
    SerializedGrootVision,
    SerializedTRTEngine,
)
from trt.vision import VisualFixedInput, nchw_to_hwc
from trt.vision_builders import build_groot_vision_export_params

GROOT_EMBODIMENT_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}

@torch.no_grad()
def make_visual_fixed_input(
    model: nn.Module,
    sample_pixel_values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> VisualFixedInput:
    eagle = model.backbone.eagle_model
    return VisualFixedInput(
        vision_model=eagle.vision_model,
        projector=eagle.mlp1,
        sample_pixel_values=sample_pixel_values,
        select_layer=eagle.select_layer,
        pixel_shuffle=eagle.use_pixel_shuffle,
        downsample_ratio=eagle.downsample_ratio,
    ).eval().to(device=device, dtype=dtype)


def make_embodiment_id(
    policy: Any,
    state: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    return torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

@torch.no_grad()
def build_context_from_language_inputs(core: nn.Module, packed: dict) -> torch.Tensor:
    eagle = core.backbone.eagle_model
    out = eagle.language_model(
        inputs_embeds=packed["inputs_embeds"],
        attention_mask=packed["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    context_embs = out.hidden_states[core.backbone.select_layer]
    context_embs = core.backbone.eagle_linear(context_embs)
    vlln_weight = getattr(core.action_head.vlln, "weight", None)
    if vlln_weight is not None:
        context_embs = context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
    context_embs = core.action_head.vlln(context_embs)
    context_embs = core.action_head.vl_self_attention(context_embs)
    return context_embs


@torch.no_grad()
def build_lm_hidden_from_language_inputs(model_or_ctx: Any, packed: dict) -> torch.Tensor:
    core = model_or_ctx.model if hasattr(model_or_ctx, "model") else model_or_ctx
    eagle = core.backbone.eagle_model
    out = eagle.language_model(
        inputs_embeds=packed["inputs_embeds"],
        attention_mask=packed["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states[core.backbone.select_layer]


@torch.no_grad()
def run_serialized_language(
    engine_lm: SerializedGrootLanguage,
    model: nn.Module,
    language_inputs: dict,
    device: torch.device,
) -> torch.Tensor:
    from trt.inference.language_prefill import build_language_prefill_inputs, run_language_prefill

    language_model = model.backbone.eagle_model.language_model
    decoder = getattr(language_model, "model", language_model)
    seq_len = int(language_inputs["inputs_embeds"].shape[1])
    prefill = build_language_prefill_inputs(
        language_inputs,
        language_model=language_model,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(language_model.config.num_key_value_heads),
        head_dim=int(language_head_dim(language_model.config)),
        max_seq_len=seq_len,
        device=device,
    )
    outputs = run_language_prefill(engine_lm, prefill, GROOT_EDGE_IO.language)
    return outputs.lm_hidden_states


@torch.no_grad()
def run_serialized_action_context(
    engine_context: SerializedGrootActionContext,
    lm_hidden_states: torch.Tensor,
) -> torch.Tensor:
    return engine_context(lm_hidden_states.to(dtype=torch.float16).contiguous())


@torch.no_grad()
def compare_edge_pipeline_to_eager(
    model: nn.Module,
    policy: Any,
    *,
    pixel_values: torch.Tensor,
    language_inputs: dict,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    trt_image_embs: torch.Tensor,
    lm_hidden_states: torch.Tensor,
    context_embs: torch.Tensor,
    trt_diffusion: nn.Module,
    device: torch.device,
    seed: int,
) -> None:
    print("\n=== Edge engine parity vs eager ===")

    with torch.no_grad():
        images_hwc = nchw_to_hwc(
            pixel_values.to(device=device, dtype=torch.float16).contiguous()
        )
        visual = make_visual_fixed_input(
            model,
            images_hwc,
            device=device,
            dtype=torch.float16,
        )
        eager_image_embs = visual(images_hwc)

    tensor_error_metrics(
        "vision",
        trt_image_embs.to(device=device, dtype=torch.float16),
        eager_image_embs.to(device=device, dtype=torch.float16),
    )

    with torch.no_grad():
        eagle = model.backbone.eagle_model
        out = eagle.language_model(
            inputs_embeds=language_inputs["inputs_embeds"],
            attention_mask=language_inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        eager_lm_hidden = out.hidden_states[model.backbone.select_layer]

    tensor_error_metrics(
        "language lm_hidden_states",
        lm_hidden_states.to(device=device, dtype=torch.float16),
        eager_lm_hidden.to(device=device, dtype=torch.float16),
    )

    with torch.no_grad():
        eager_context_embs = build_context_from_language_inputs(model, language_inputs)

    tensor_error_metrics(
        "action_context vl_embs",
        context_embs.to(device=device, dtype=torch.float16),
        eager_context_embs.to(device=device, dtype=torch.float16),
    )

    state_fp16 = state.to(device=device, dtype=torch.float16).contiguous()
    embodiment_id = embodiment_id.to(device=device).contiguous()
    context_fp16 = context_embs.to(device=device, dtype=torch.float16).contiguous()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(
        context_fp16.shape[0],
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=context_fp16.dtype,
        generator=generator,
    )
    timestep = torch.zeros(context_fp16.shape[0], device=device, dtype=torch.long)

    eager_action_module = make_groot_static_action_module(
        model.action_head,
        device,
        torch.float16,
        embodiment_id,
    )
    with torch.no_grad():
        eager_velocity = eager_action_module(
            noise,
            timestep,
            context_fp16,
            state_fp16,
            embodiment_id,
        )
        trt_velocity = trt_diffusion(
            noise,
            timestep,
            context_fp16,
            state_fp16,
            embodiment_id,
        )
    tensor_error_metrics(
        "diffusion velocity",
        trt_velocity.to(device=device, dtype=torch.float16),
        eager_velocity.to(device=device, dtype=torch.float16),
    )

    with torch.no_grad():
        eager_actions = sample_actions_raw(
            eager_action_module,
            ActionRolloutContext(
                noise=noise,
                device=device,
                context_embs=context_fp16,
                state=state_fp16,
                embodiment_id=embodiment_id,
            ),
            GROOTActionAdapter(model.action_head),
        )
        trt_actions = sample_actions_raw(
            trt_diffusion,
            ActionRolloutContext(
                noise=noise,
                device=device,
                context_embs=context_fp16,
                state=state_fp16,
                embodiment_id=embodiment_id,
            ),
            GROOTActionAdapter(model.action_head),
        )
    action_metrics = compute_policy_action_metrics(
        trt_actions,
        eager_actions,
        policy,
    )
    print(
        "full action rollout:",
        f"action_ade={action_metrics['action_ade']:.6f}",
        f"mean_abs={action_metrics['mean_abs']:.6f}",
    )

def compute_policy_action_metrics(
    trt_actions: torch.Tensor,
    eager_actions: torch.Tensor,
    policy: Any,
) -> dict[str, float]:
    output_features = getattr(policy.config, "output_features", None)
    action_dim = None
    if output_features is not None:
        from lerobot.utils.constants import ACTION

        action_feature = output_features.get(ACTION)
        if action_feature is not None:
            shape = getattr(action_feature, "shape", None)
            if shape:
                action_dim = int(shape[0])
    if action_dim is not None:
        trt_actions = trt_actions[..., :action_dim]
        eager_actions = eager_actions[..., :action_dim]
    return compute_action_parity_metrics(trt_actions, eager_actions)

def dump_edge_fixture(ctx: ExportContext) -> pathlib.Path:
    model = ctx.model
    policy = ctx.policy
    device = ctx.device
    language_inputs = ctx.language_inputs
    input_ids = ctx.tokenized["input_ids"]
    pixel_values = ctx.pixel_values
    visual_embeds = ctx.image_embs
    context_embs = ctx.context_embs
    state = ctx.action_side["state"]
    embodiment_id = ctx.action_side["embodiment_id"]

    eagle = model.backbone.eagle_model
    text_embeds = eagle.language_model.get_input_embeddings()(
        input_ids.to(device=device)
    ).to(device=device, dtype=torch.float16)

    state = state.to(device=device, dtype=torch.float16).contiguous()
    context_embs = context_embs.to(device=device, dtype=torch.float16).contiguous()
    embodiment_id = embodiment_id.to(device=device).contiguous()

    generator = torch.Generator(device=device)
    generator.manual_seed(ctx.seed)
    initial_actions = torch.randn(
        context_embs.shape[0],
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=context_embs.dtype,
        generator=generator,
    )
    timestep = torch.zeros(context_embs.shape[0], device=device, dtype=torch.long)

    action_module = make_groot_static_action_module(
        model.action_head,
        device,
        torch.float16,
        embodiment_id,
    )
    velocity = action_module(
        initial_actions,
        timestep,
        context_embs,
        state,
        embodiment_id,
    )
    actions_out = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=initial_actions,
            device=device,
            context_embs=context_embs,
            state=state,
            embodiment_id=embodiment_id,
        ),
        GROOTActionAdapter(model.action_head),
    )

    if language_inputs.get("image_token_mask") is None:
        raise ValueError("GR00T Edge fixture export requires image_token_mask")

    return write_edge_fixture(
        str(ctx.engine_root),
        {
            "pixel_values": pixel_values.to(device=device, dtype=torch.float16),
            "text_embeds": text_embeds,
            "image_token_mask": language_inputs["image_token_mask"].to(dtype=torch.uint8),
            "inputs_embeds": language_inputs["inputs_embeds"].to(device=device, dtype=torch.float16),
            "visual_embeds": visual_embeds.to(device=device, dtype=torch.float16),
            "context_embs": context_embs,
            "state": state,
            "embodiment_id": embodiment_id,
            "initial_actions": initial_actions,
            "timestep": timestep,
            "velocity": velocity.to(device=device, dtype=torch.float16),
            "actions_out": actions_out.to(device=device, dtype=torch.float16),
        },
    )
class GrootExportHooks(VLAExportHooks):
    def __init__(
        self,
        *,
        io: PipelineIOSpec,
        tokenizer: Any | None = None,
        vision_trt_settings: dict | None = None,
        action_trt_settings: dict | None = None,
    ) -> None:
        self.io = io
        self.tokenizer = tokenizer
        self.vision_trt_settings = vision_trt_settings or dict(VISION_TRT_SETTINGS)
        self.action_trt_settings = action_trt_settings or dict(ACTION_TRT_SETTINGS)

    def has_action_context(self, ctx: ExportContext) -> bool:
        if self.io.action_context is None:
            return False
        return True

    def preprocess(self, ctx: ExportContext) -> None:
        tokenized_data = ctx.model_inputs["tokenized_data"]
        ctx.tokenized = {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        ctx.pixel_values = tokenized_data["pixel_values"].to(
            device=ctx.device,
            dtype=torch.float16,
        )
        state = pack_state(
            ctx.model_inputs["state"],
            max_state_dim=ctx.policy.config.max_state_dim,
            device=ctx.device,
        )
        ctx.action_side = {
            "state": state,
            "embodiment_id": make_embodiment_id(ctx.policy, state, ctx.device),
        }

    def build_vision_spec(self, ctx: ExportContext):
        return build_groot_vision_export_params(
            ctx.model,
            ctx.pixel_values,
            ctx.device,
            io=self.io,
            trt_settings=self.vision_trt_settings,
            input_dtype=torch.float16,
        )

    def pack_language_inputs(self, ctx: ExportContext) -> dict:
        return pack_groot_language_inputs(
            ctx.model,
            ctx.image_embs,
            ctx.tokenized["input_ids"],
            ctx.tokenized["attention_mask"],
        )

    def build_language_spec(self, ctx: ExportContext):
        if ctx.vis_spec is None:
            raise RuntimeError("vis_spec required before build_language_spec")
        return build_groot_language_export_params(
            ctx.model,
            ctx.tokenized["input_ids"],
            image_token_id=int(ctx.vis_spec.image_token_id),
            seq_len_per_image=int(ctx.vis_spec.config_seq_len),
            device=ctx.device,
            io=self.io,
            dtype=torch.float16,
        )

    def build_chat_template(self, tokenizer: Any) -> dict[str, Any]:
        """Build processed_chat_template.json for VitRunner (single image placeholder per image)."""
        im_start = "<|im_start|>"
        im_end = tokenizer.eos_token
        if not im_end:
            raise ValueError("Tokenizer eos_token is required to build the GROOT chat template.")

        system_only = tokenizer.apply_chat_template(
            [{"role": "system", "content": "SYS"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": "TEXTONLY"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        with_gen = tokenizer.apply_chat_template(
            [{"role": "user", "content": "TEXTONLY"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        system_prefix = system_only.split("SYS", 1)[0]
        system_suffix = "SYS" + system_only.split("SYS", 1)[1]

        user_prefix = user_only.split("TEXTONLY", 1)[0]
        user_suffix = "TEXTONLY" + user_only.split("TEXTONLY", 1)[1]

        assistant_only = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "TEXTONLY"},
                {"role": "assistant", "content": "ASSIST"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        assistant_prefix = assistant_only[len(user_only) :].split("ASSIST", 1)[0]
        assistant_suffix = "ASSIST" + assistant_only.split("ASSIST", 1)[1]

        generation_prompt = with_gen[len(user_only) :]

        return {
            "model_path": "groot-vitrunner",
            "roles": {
                "system": {"prefix": system_prefix, "suffix": system_suffix},
                "user": {"prefix": user_prefix, "suffix": user_suffix},
                "assistant": {"prefix": assistant_prefix, "suffix": assistant_suffix},
            },
            "content_types": {
                "image": {"format": "<img><IMG_CONTEXT></img>"},
            },
            "generation_prompt": generation_prompt,
            "default_system_prompt": "You are a helpful assistant.",
        }

    def build_action_context(self, ctx: ExportContext) -> ComponentBuild:
        lm_hidden = ctx.lm_hidden_states
        if lm_hidden is None:
            raise RuntimeError("lm_hidden_states required for action_context export")
        module = make_action_context_module(
            ctx.model,
            device=ctx.device,
            dtype=torch.float16,
        )
        lm_hidden = lm_hidden.to(device=ctx.device, dtype=torch.float16).contiguous()
        with torch.no_grad():
            eager_output = module(lm_hidden)
        return ComponentBuild(
            module=module,
            sample_inputs=(lm_hidden,),
            extra_config={
                "engine_role": "preprocess_action_input",
                "context_seq_len": int(lm_hidden.shape[1]),
                "context_hidden_size": int(eager_output.shape[2]),
            },
            trt_settings=self.action_trt_settings,
            model_type="action_context",
            component="action_context",
            engine_file="action_context.engine",
        )

    def build_diffusion_spec(self, ctx: ExportContext):
        if ctx.context_embs is None:
            raise RuntimeError("context_embs required before build_diffusion_spec")
        return build_groot_diffusion_export_params(
            ctx.model,
            context_embs=ctx.context_embs,
            state=ctx.action_side["state"],
            embodiment_id=ctx.action_side["embodiment_id"],
            device=ctx.device,
            io=self.io,
            trt_settings=self.action_trt_settings,
        )

    def after_export(self, ctx: ExportContext, sink: ExportSink) -> None:
        if not ctx.accuracy_check:
            return

        if sink.mode is ExportMode.SERIALIZED:
            language_dir = ctx.engine_subdir("language")
            action_context_dir = ctx.engine_subdir("action_context")
            action_dir = ctx.engine_subdir("action")
            vision_dir = ctx.engine_subdir("visual")

            if ctx.accuracy_check:
                vision_runner = SerializedGrootVision(SerializedTRTEngine(vision_dir))
                language_runner = SerializedGrootLanguage(SerializedTRTEngine(language_dir))
                with torch.no_grad():
                    trt_image_embs = vision_runner(ctx.pixel_values)
                    ctx.lm_hidden_states = run_serialized_language(
                        language_runner,
                        ctx.model,
                        ctx.language_inputs,
                        ctx.device,
                    )

                action_context_runner = SerializedGrootActionContext(
                    SerializedTRTEngine(action_context_dir)
                )
                with torch.no_grad():
                    ctx.context_embs = run_serialized_action_context(
                        action_context_runner,
                        ctx.lm_hidden_states,
                    )

                action_runner = SerializedGrootAction(SerializedTRTEngine(action_dir))
                ctx.handles["action_runner"] = action_runner

                compare_edge_pipeline_to_eager(
                    ctx.model,
                    ctx.policy,
                    pixel_values=ctx.pixel_values,
                    language_inputs=ctx.language_inputs,
                    state=ctx.action_side["state"],
                    embodiment_id=ctx.action_side["embodiment_id"],
                    trt_image_embs=trt_image_embs,
                    lm_hidden_states=ctx.lm_hidden_states,
                    context_embs=ctx.context_embs,
                    trt_diffusion=action_runner,
                    device=ctx.device,
                    seed=ctx.seed,
                )

            if ctx.engine_root is not None:
                dump_edge_fixture(ctx)
            return

        if ctx.context_embs is not None and ctx.image_embs is not None:
            from trt.measure import tensor_error_metrics as _tensor_error_metrics
            from trt.vision import nchw_to_hwc

            with torch.no_grad():
                images_hwc = nchw_to_hwc(
                    ctx.pixel_values.to(device=ctx.device, dtype=torch.float16).contiguous()
                )
                visual = make_visual_fixed_input(
                    ctx.model,
                    images_hwc,
                    device=ctx.device,
                    dtype=torch.float16,
                )
                eager_image_embs = visual(images_hwc)
            _tensor_error_metrics(
                "groot TRT vs original vision embeddings",
                ctx.image_embs.to(device=ctx.device, dtype=torch.float16),
                eager_image_embs.to(device=ctx.device, dtype=torch.float16),
            )

            with torch.no_grad():
                eager_context = build_context_from_language_inputs(
                    ctx.model,
                    ctx.language_inputs,
                )
            _tensor_error_metrics(
                "groot TRT vs eager language context (TRT vision)",
                ctx.context_embs,
                eager_context.to(device=ctx.device, dtype=torch.float16),
            )
