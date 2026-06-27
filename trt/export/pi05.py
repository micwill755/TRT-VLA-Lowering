"""PI0.5-specific export hooks and helpers."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from lerobot.utils.constants import ACTION

from trt.action_rollout import ActionRolloutContext, PrefixKVFlowActionAdapter, sample_actions_raw
from trt.chat_template import build_pi05_vitrunner_chat_template
from trt.compile import dump_edge_fixture
from trt.diffusion_builders import (
    build_pi05_diffusion_export_params,
    make_pi05_action_compile_inputs,
    make_pi05_static_action_module,
)
from trt.export.context import ExportContext
from trt.export.hooks import VLAExportHooks
from trt.export.mode import ExportMode
from trt.export.settings import ACTION_TRT_SETTINGS, VISION_TRT_SETTINGS
from trt.export.sinks import ExportSink
from trt.io_spec import PI05_EDGE_IO, PipelineIOSpec
from trt.language import (
    language_head_dim,
    pi05_plugin_lm_smoke_check,
    run_prefix_language_eager,
)
from trt.language_builders import build_pi05_language_export_params
from trt.measure import compute_action_parity_metrics, tensor_error_metrics
from trt.packing import pack_pi05_prefix
from trt.rope import make_rope_rotary_cos_sin
from trt.serialize import SerializedPI05Action, SerializedPI05Language, SerializedTRTEngine
from trt.utils import ensure_pi05_paligemma_on_device, make_suffix_position_and_mask, prepare_policy_inputs
from trt.vision import VIT_ENGINE_INPUT_NAME, nchw_to_hwc
from trt.vision_builders import build_pi05_vision_export_params

PALIGEMMA_TOKENIZER_ID = "google/paligemma-3b-pt-224"


def configure_torch_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def action_output_dim(policy: Any) -> int:
    output_feature = policy.config.output_features.get(ACTION)
    if output_feature is None:
        return int(policy.model.config.max_action_dim)
    return int(output_feature.shape[0])


def crop_policy_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    return actions[..., : action_output_dim(policy)]


def validate_language_len(prefix: dict, max_seq_len: int | None) -> int:
    prefix_len = int(prefix["inputs_embeds"].shape[1])
    if max_seq_len is not None and int(max_seq_len) != prefix_len:
        raise ValueError(
            "PI0.5 Edge export uses a compact static prefix. "
            f"--max-seq-len must match compact prefix length {prefix_len}, got {max_seq_len}."
        )
    return prefix_len


def _tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu()
    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    return Image.fromarray(img.numpy())


def write_pi05_runtime_smoke_case(
    engine_root: str | pathlib.Path,
    *,
    task_text: str,
    image: torch.Tensor,
    max_generate_length: int = 0,
) -> pathlib.Path:
    engine_root = pathlib.Path(engine_root)
    smoke_dir = engine_root / "runtime_smoke"
    image_path = smoke_dir / "camera_0.png"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _tensor_image_to_pil(image).save(image_path)

    payload = {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": int(max_generate_length),
        "requests": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path.resolve())},
                            {"type": "text", "text": task_text},
                        ],
                    }
                ],
            }
        ],
    }
    input_path = smoke_dir / "input.json"
    input_path.write_text(json.dumps(payload, indent=2) + "\n")
    return input_path


class PI05VisionEngineAdapter:
    """Serialized VitRunner wrapper: NCHW policy tensors -> HWC engine binding ``input``."""

    def __init__(self, engine: SerializedTRTEngine):
        self.engine = engine

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hwc = nchw_to_hwc(pixel_values.to(device=pixel_values.device).contiguous())
        return self.engine({VIT_ENGINE_INPUT_NAME: hwc})[0]


def run_serialized_pi05_language(
    language_runner: SerializedPI05Language,
    prefix_embs: torch.Tensor,
    *,
    max_seq_len: int,
    device: torch.device,
    position_ids: torch.Tensor | None,
    cfg,
    language_model=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_embs = prefix_embs.to(device=device, dtype=torch.float16).contiguous()
    batch_size = int(prefix_embs.shape[0])
    kv_caches = [
        torch.zeros(
            batch_size,
            2,
            int(cfg.num_key_value_heads),
            int(max_seq_len),
            language_head_dim(cfg),
            device=device,
            dtype=prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]
    ctx_len = torch.full((batch_size,), prefix_embs.shape[1], device=device, dtype=torch.int32)
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        int(max_seq_len),
        device,
        language_model=language_model,
        position_ids=position_ids,
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full(
        (batch_size, 1),
        int(prefix_embs.shape[1]) - 1,
        device=device,
        dtype=torch.int64,
    )
    return language_runner(
        prefix_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        kv_caches,
    )


@torch.no_grad()
def compare_pi05_edge_pipeline_to_eager(
    core,
    policy,
    *,
    images: list[torch.Tensor],
    img_masks,
    tokens,
    masks,
    trt_image_embs: torch.Tensor | list[torch.Tensor],
    trt_hidden: torch.Tensor,
    trt_prefix_k: torch.Tensor,
    trt_prefix_v: torch.Tensor,
    trt_diffusion: nn.Module,
    device: torch.device,
    seed: int,
) -> None:
    print("\n=== PI0.5 Edge engine parity vs eager ===")

    ensure_pi05_paligemma_on_device(core, device)
    eager_image_embs = [core.paligemma_with_expert.embed_image(image) for image in images]
    trt_embs = trt_image_embs if isinstance(trt_image_embs, list) else [trt_image_embs]
    for idx, (trt_emb, eager_emb) in enumerate(zip(trt_embs, eager_image_embs)):
        tensor_error_metrics(
            f"vision[{idx}]",
            trt_emb.to(device=device, dtype=torch.float16),
            eager_emb.to(device=device, dtype=torch.float16),
        )

    compact_prefix = pack_pi05_prefix(core, eager_image_embs, img_masks, tokens, masks)
    eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
        core.paligemma_with_expert.paligemma.model.language_model,
        compact_prefix["inputs_embeds"],
        compact_prefix["attention_mask"],
        compact_prefix["position_ids"],
    )
    tensor_error_metrics(
        "language lm_hidden_states",
        trt_hidden.to(device=device, dtype=torch.float16),
        eager_hidden.to(device=device, dtype=torch.float16),
    )
    tensor_error_metrics(
        "language prefix_k",
        trt_prefix_k.to(device=device, dtype=torch.float16),
        eager_prefix_k.to(device=device, dtype=torch.float16),
    )
    tensor_error_metrics(
        "language prefix_v",
        trt_prefix_v.to(device=device, dtype=torch.float16),
        eager_prefix_v.to(device=device, dtype=torch.float16),
    )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    noise = core.sample_noise(
        (tokens.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )
    position_ids, attention_mask = make_suffix_position_and_mask(
        core,
        compact_prefix["pad_mask"],
        noise,
        device,
    )
    prefix_k = trt_prefix_k.to(device=device, dtype=noise.dtype).contiguous()
    prefix_v = trt_prefix_v.to(device=device, dtype=noise.dtype).contiguous()
    timestep = torch.full((tokens.shape[0],), 1.0, dtype=torch.float32, device=device)

    eager_action_module = make_pi05_static_action_module(core, device)
    with torch.no_grad():
        eager_velocity = eager_action_module(
            noise,
            timestep,
            prefix_k,
            prefix_v,
            position_ids.contiguous(),
            attention_mask.contiguous(),
        )
        trt_velocity = trt_diffusion(
            noise,
            timestep,
            prefix_k,
            prefix_v,
            position_ids.contiguous(),
            attention_mask.contiguous(),
        )
    tensor_error_metrics("diffusion velocity", trt_velocity, eager_velocity)

    adapter = PrefixKVFlowActionAdapter(core, int(core.config.num_inference_steps))
    eager_actions = sample_actions_raw(
        eager_action_module,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=compact_prefix["pad_mask"],
        ),
        adapter,
    )
    trt_actions = sample_actions_raw(
        trt_diffusion,
        ActionRolloutContext(
            noise=noise,
            device=device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=compact_prefix["pad_mask"],
        ),
        adapter,
    )
    metrics = compute_action_parity_metrics(
        crop_policy_actions(policy, trt_actions),
        crop_policy_actions(policy, eager_actions),
    )
    print(
        f"full rollout action_ade={metrics['action_ade']:.6f}  "
        f"mean_abs={metrics['mean_abs']:.6f}"
    )


@torch.no_grad()
def dump_pi05_edge_fixture(
    ctx: ExportContext,
    *,
    lm_hidden_states: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
) -> pathlib.Path:
    core = ctx.model
    compact_prefix = ctx.language_inputs
    batch_size = int(compact_prefix["inputs_embeds"].shape[0])
    noise = core.sample_noise(
        (batch_size, core.config.chunk_size, core.config.max_action_dim),
        ctx.device,
    )
    position_ids, attention_mask = make_suffix_position_and_mask(
        core,
        compact_prefix["pad_mask"],
        noise,
        ctx.device,
    )
    num_steps = int(core.config.num_inference_steps)
    timestep = torch.full((batch_size,), 1.0, dtype=torch.float32, device=ctx.device)
    action_module = make_pi05_static_action_module(core, ctx.device)
    prefix_k = prefix_k.to(device=ctx.device, dtype=noise.dtype).contiguous()
    prefix_v = prefix_v.to(device=ctx.device, dtype=noise.dtype).contiguous()
    velocity = action_module(
        noise,
        timestep,
        prefix_k,
        prefix_v,
        position_ids.contiguous(),
        attention_mask.contiguous(),
    )
    actions_out = sample_actions_raw(
        action_module,
        ActionRolloutContext(
            noise=noise,
            device=ctx.device,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_pad_mask=compact_prefix["pad_mask"],
        ),
        PrefixKVFlowActionAdapter(core, num_steps),
    )
    velocity_name = ctx.io.action.output_names[0]
    return dump_edge_fixture(
        str(ctx.engine_root),
        {
            "pixel_values": ctx.pixel_values.to(device=ctx.device).contiguous(),
            "inputs_embeds": compact_prefix["inputs_embeds"].to(device=ctx.device, dtype=torch.float16),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "lm_hidden_states": lm_hidden_states.to(device=ctx.device, dtype=torch.float16),
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "initial_actions": noise,
            "timestep": timestep,
            velocity_name: velocity,
            "actions_out": actions_out.to(device=ctx.device, dtype=noise.dtype),
        },
    )


class Pi05ExportHooks(VLAExportHooks):
    def __init__(
        self,
        *,
        io: PipelineIOSpec = PI05_EDGE_IO,
        tokenizer: Any | None = None,
        vision_trt_settings: dict | None = None,
        action_trt_settings: dict | None = None,
        stage_parity: bool = True,
        max_generate_length: int = 0,
    ) -> None:
        self.io = io
        self.tokenizer = tokenizer
        self.vision_trt_settings = vision_trt_settings or dict(VISION_TRT_SETTINGS)
        self.action_trt_settings = action_trt_settings or dict(ACTION_TRT_SETTINGS)
        self.stage_parity = stage_parity
        self.max_generate_length = max_generate_length

    def preprocess(self, ctx: ExportContext) -> None:
        images, img_masks, tokens, masks = prepare_policy_inputs(
            ctx.policy,
            ctx.model_inputs,
            ctx.device,
        )
        ctx.pixel_values = images[0].to(device=ctx.device).contiguous()
        ctx.action_side = {
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "batch_size": int(tokens.shape[0]),
        }
        ensure_pi05_paligemma_on_device(ctx.model, ctx.device)

    def build_vision_spec(self, ctx: ExportContext):
        return build_pi05_vision_export_params(
            ctx.model,
            ctx.pixel_values,
            ctx.device,
            io=self.io,
            trt_settings=self.vision_trt_settings,
        )

    def pack_language_inputs(self, ctx: ExportContext) -> dict:
        return pack_pi05_prefix(
            ctx.model,
            ctx.image_embs,
            ctx.action_side["img_masks"],
            ctx.action_side["tokens"],
            ctx.action_side["masks"],
            inputs_dtype=torch.float16,
        )

    def build_language_spec(self, ctx: ExportContext):
        validate_language_len(ctx.language_inputs, ctx.max_seq_len)
        return build_pi05_language_export_params(
            ctx.model,
            ctx.language_inputs,
            ctx.device,
            io=self.io,
            trt_settings=self.action_trt_settings,
        )

    def build_chat_template(self, tokenizer: Any) -> dict[str, Any]:
        """Minimal processed_chat_template.json for VitRunner image placeholder expansion."""
        return {
            "model_path": "pi05-vitrunner",
            "roles": {
                "user": {"prefix": "", "suffix": ""},
            },
            "content_types": {
                "image": {"format": image_format},
            },
            "generation_prompt": "",
            "default_system_prompt": "",
        }

    def build_diffusion_spec(self, ctx: ExportContext):
        pad_mask = ctx.language_inputs["pad_mask"]
        return build_pi05_diffusion_export_params(
            ctx.model,
            batch_size=int(ctx.action_side["batch_size"]),
            prefix_len=int(pad_mask.shape[1]),
            device=ctx.device,
            io=self.io,
            trt_settings=self.action_trt_settings,
        )

    def after_export(self, ctx: ExportContext, sink: ExportSink) -> None:
        if not ctx.accuracy_check:
            return

        if sink.mode is ExportMode.SERIALIZED:
            if not self.stage_parity:
                return

            language_dir = ctx.engine_subdir("language")
            action_dir = ctx.engine_subdir("action")
            language_runner = SerializedPI05Language(SerializedTRTEngine(language_dir))
            lm_cfg = ctx.model.paligemma_with_expert.paligemma.model.language_model.config
            language_model = ctx.model.paligemma_with_expert.paligemma.model.language_model
            max_seq_len = int(ctx.lang_spec.max_seq_len)

            with torch.no_grad():
                trt_hidden, trt_prefix_k, trt_prefix_v = run_serialized_pi05_language(
                    language_runner,
                    ctx.language_inputs["inputs_embeds"],
                    max_seq_len=max_seq_len,
                    device=ctx.device,
                    position_ids=ctx.language_inputs.get("position_ids"),
                    cfg=lm_cfg,
                    language_model=language_model,
                )

            action_runner = SerializedPI05Action(SerializedTRTEngine(action_dir))
            compare_pi05_edge_pipeline_to_eager(
                ctx.model,
                ctx.policy,
                images=ctx.action_side["images"],
                img_masks=ctx.action_side["img_masks"],
                tokens=ctx.action_side["tokens"],
                masks=ctx.action_side["masks"],
                trt_image_embs=ctx.image_embs,
                trt_hidden=trt_hidden,
                trt_prefix_k=trt_prefix_k,
                trt_prefix_v=trt_prefix_v,
                trt_diffusion=action_runner,
                device=ctx.device,
                seed=ctx.seed,
            )

            with torch.no_grad():
                eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
                    language_model,
                    ctx.language_inputs["inputs_embeds"],
                    ctx.language_inputs["attention_mask"],
                    ctx.language_inputs["position_ids"],
                )

            if ctx.engine_root is not None:
                fixture_dir = dump_pi05_edge_fixture(
                    ctx,
                    lm_hidden_states=eager_hidden,
                    prefix_k=eager_prefix_k,
                    prefix_v=eager_prefix_v,
                )
                ctx.plugin_info["fixture_dir"] = str(fixture_dir)

                task_text = ctx.model_inputs.get("task", "")
                if isinstance(task_text, (list, tuple)):
                    task_text = task_text[0] if task_text else "pick up the object"
                smoke_input = write_pi05_runtime_smoke_case(
                    ctx.engine_root,
                    task_text=str(task_text),
                    image=ctx.pixel_values,
                    max_generate_length=self.max_generate_length,
                )
                ctx.plugin_info["runtime_smoke_input"] = str(smoke_input)
            return

        if ctx.handles.get("language") is not None:
            pi05_plugin_lm_smoke_check(
                ctx.model,
                ctx.handles["language"],
                ctx.language_inputs["inputs_embeds"],
                max_seq_len=int(ctx.lang_spec.max_seq_len),
                device=ctx.device,
                attention_mask=ctx.language_inputs["attention_mask"],
                position_ids=ctx.language_inputs["position_ids"],
                prefix_pad_masks=ctx.language_inputs["pad_mask"],
                max_logit_tokens=16,
            )

    def finalize_plugin_info(self, ctx: ExportContext) -> dict:
        info = super().finalize_plugin_info(ctx)
        pad_mask = ctx.language_inputs.get("pad_mask")
        if pad_mask is not None:
            info["prefix_seq_len"] = int(pad_mask.shape[1])
        info["chunk_size"] = int(ctx.model.config.chunk_size)
        info["max_action_dim"] = int(ctx.model.config.max_action_dim)
        info["output_action_dim"] = action_output_dim(ctx.policy)
        info["num_inference_steps"] = int(ctx.model.config.num_inference_steps)
        if ctx.lang_spec is not None:
            info["language_max_seq_len"] = int(ctx.lang_spec.max_seq_len)
        root = ctx.engine_root
        if root is not None:
            info.update(
                {
                    "vision_engine_dir": str(root / "visual"),
                    "language_engine_dir": str(root / "language"),
                    "action_engine_dir": str(root / "action"),
                    "vision_engine": str(root / "visual" / "visual.engine"),
                    "language_engine": str(root / "language" / "language.engine"),
                    "diffusion_engine": str(root / "action" / "diffusion.engine"),
                }
            )
        return info
