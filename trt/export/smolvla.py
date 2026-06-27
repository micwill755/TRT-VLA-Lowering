"""SmolVLA-specific export hooks and helpers."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from trt.action_rollout import ActionRolloutContext, PrefixKVFlowActionAdapter, sample_actions_raw
from trt.chat_template import build_smolvla_vitrunner_chat_template
from trt.compile import dump_edge_fixture
from trt.diffusion_builders import build_smolvla_diffusion_export_params
from trt.edge_llm_runtime import write_llm_runtime_smoke_case
from trt.export.context import ExportContext
from trt.export.hooks import VLAExportHooks
from trt.export.mode import ExportMode
from trt.export.settings import ACTION_TRT_SETTINGS, VISION_TRT_SETTINGS
from trt.export.sinks import ExportSink
from trt.io_spec import PI05_EDGE_IO, PipelineIOSpec
from trt.language import language_head_dim, make_plugin_lm_causal_wrapper
from trt.language_builders import build_smolvla_language_export_params
from trt.measure import compare_language, compute_action_parity_metrics, tensor_error_metrics
from trt.packing import pack_smolvla_prefix
from trt.rope import make_rope_rotary_cos_sin
from trt.serialize import SerializedPI05Action, SerializedPI05Language, SerializedTRTEngine
from trt.utils import ensure_smolvla_on_device, make_smolvla_runner_inputs
from trt.vision import VIT_ENGINE_INPUT_NAME, nchw_to_hwc
from trt.vision_builders import build_smolvla_vision_export_params


def action_output_dim(policy: Any) -> int:
    output_feature = policy.config.output_features.get(ACTION)
    if output_feature is None:
        return int(policy.model.config.max_action_dim)
    return int(output_feature.shape[0])


def validate_language_len(prefix: dict, max_seq_len: int | None) -> int:
    prefix_len = int(prefix["inputs_embeds"].shape[1])
    if max_seq_len is not None and int(max_seq_len) != prefix_len:
        raise ValueError(
            "SmolVLA Edge export uses a static prefix. "
            f"--max-seq-len must match prefix length {prefix_len}, got {max_seq_len}."
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


def _smolvla_task_text(batch: dict[str, Any]) -> str:
    task_text = batch.get("task", "")
    if isinstance(task_text, (list, tuple)):
        task_text = task_text[0] if task_text else "pick up the object"
    task_text = str(task_text)
    if task_text and not task_text.endswith("\n"):
        task_text += "\n"
    return task_text


@torch.no_grad()
def prepare_smolvla_batch(policy, batch, device: torch.device):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
    images = [img.to(device=device, dtype=torch.float32) for img in images]
    img_masks = [mask.to(device=device) for mask in img_masks]
    state = state.to(device=device, dtype=torch.float32)
    return images, img_masks, tokens, masks, state


def smolvla_text_config(core):
    return core.vlm_with_expert.get_vlm_model().config.text_config


def smolvla_language_model(core):
    return core.vlm_with_expert.get_vlm_model().text_model


def smolvla_action_adapter(core) -> PrefixKVFlowActionAdapter:
    return PrefixKVFlowActionAdapter(
        core,
        int(core.config.num_steps),
        runner_inputs_fn=make_smolvla_runner_inputs,
    )


class SmolVLAVisionEngineAdapter:
    """Serialized VitRunner wrapper: NCHW policy tensors -> [B, S, H] image embeds."""

    def __init__(self, engine: SerializedTRTEngine, *, num_tokens: int):
        self.engine = engine
        self.num_tokens = int(num_tokens)

    def __call__(self, pixel_values_nchw: torch.Tensor) -> torch.Tensor:
        hwc = nchw_to_hwc(pixel_values_nchw.contiguous())
        flat = self.engine({VIT_ENGINE_INPUT_NAME: hwc})[0]
        batch_size = int(pixel_values_nchw.shape[0])
        hidden = flat.shape[-1]
        expected_rows = batch_size * self.num_tokens
        if flat.shape[0] != expected_rows:
            raise ValueError(
                f"Visual engine returned {tuple(flat.shape)} rows, expected {expected_rows}"
            )
        return flat.reshape(batch_size, self.num_tokens, hidden)


class SerializedSmolVLAVision:
    def __init__(self, engine: SerializedTRTEngine):
        builder_config = engine.config.get("builder_config", {})
        seq_len = builder_config.get("seq_len", engine.config.get("seq_len"))
        if seq_len is None:
            raise KeyError(f"Missing vision seq_len in engine config: {engine.engine_dir}")
        self._adapter = SmolVLAVisionEngineAdapter(engine, num_tokens=int(seq_len))

    def __call__(self, pixel_values_nchw: torch.Tensor) -> torch.Tensor:
        return self._adapter(pixel_values_nchw)


def run_serialized_smolvla_language(
    language_runner: SerializedPI05Language,
    prefix_embs: torch.Tensor,
    *,
    max_seq_len: int,
    num_layers: int,
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
        for _ in range(int(num_layers))
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
def run_smolvla_plugin_language_eager(
    core,
    prefix: dict,
    device: torch.device,
    *,
    language_model=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_embs = prefix["inputs_embeds"].to(device=device, dtype=torch.float16).contiguous()
    max_seq_len = int(prefix_embs.shape[1])
    batch_size = int(prefix_embs.shape[0])
    cfg = smolvla_text_config(core)
    num_layers = int(core.vlm_with_expert.num_vlm_layers)

    lm = smolvla_language_model(core)
    decoder = getattr(lm, "model", lm)
    lm_head = core.vlm_with_expert.vlm.lm_head
    lm_wrapper = make_plugin_lm_causal_wrapper(
        decoder,
        lm_head,
        hidden_size=int(cfg.hidden_size),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        log_prefix="smolvla",
    ).to(device=device, dtype=torch.float16).eval()

    kv_caches = [
        torch.zeros(
            batch_size,
            2,
            int(cfg.num_key_value_heads),
            max_seq_len,
            language_head_dim(cfg),
            device=device,
            dtype=prefix_embs.dtype,
        )
        for _ in range(num_layers)
    ]
    ctx_len = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.int32)
    rope_rotary_cos_sin = make_rope_rotary_cos_sin(
        cfg,
        max_seq_len,
        device,
        language_model=language_model,
        position_ids=prefix["position_ids"],
    )
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full((batch_size, 1), max_seq_len - 1, device=device, dtype=torch.int64)

    _, hidden, prefix_k, prefix_v = lm_wrapper(
        prefix_embs,
        rope_rotary_cos_sin,
        ctx_len,
        kvcache_start_index,
        last_token_ids,
        *kv_caches,
    )
    return hidden, prefix_k, prefix_v


@torch.no_grad()
def compare_smolvla_edge_pipeline_to_eager(
    *,
    core,
    policy,
    images,
    img_masks,
    tokens,
    masks,
    state,
    trt_image_embs,
    trt_hidden: torch.Tensor,
    trt_prefix_k: torch.Tensor,
    trt_prefix_v: torch.Tensor,
    action_runner: SerializedPI05Action,
    device: torch.device,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print("\n=== SmolVLA Edge engine parity vs eager ===")

    ensure_smolvla_on_device(core, device)
    eager_image_embs = [core.vlm_with_expert.embed_image(img) for img in images]
    eager_prefix = pack_smolvla_prefix(core, eager_image_embs, img_masks, tokens, masks, state)
    eager_hidden, eager_prefix_k, eager_prefix_v = run_smolvla_plugin_language_eager(
        core,
        eager_prefix,
        device,
        language_model=smolvla_language_model(core),
    )

    for i, (eager_img, trt_img) in enumerate(zip(eager_image_embs, trt_image_embs, strict=True)):
        tensor_error_metrics(f"vision[{i}]", trt_img, eager_img)

    compare_language(
        eager_hidden,
        eager_prefix_k,
        eager_prefix_v,
        trt_hidden,
        trt_prefix_k,
        trt_prefix_v,
        eager_prefix["pad_mask"],
    )

    noise = core.sample_noise(
        (state.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )
    eager_actions = core.sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        state,
        noise=noise.clone(),
    )
    trt_actions = sample_actions_raw(
        action_runner,
        ActionRolloutContext(
            noise=noise.clone(),
            device=device,
            prefix_k=trt_prefix_k,
            prefix_v=trt_prefix_v,
            prefix_pad_mask=eager_prefix["pad_mask"],
        ),
        smolvla_action_adapter(core),
    )

    action_dim = action_output_dim(policy)
    metrics = compute_action_parity_metrics(
        trt_actions[:, :, :action_dim],
        eager_actions[:, :, :action_dim],
    )
    print(
        f"  action parity: ADE={metrics['action_ade']:.6f}  "
        f"FDE={metrics['action_fde']:.6f}  "
        f"mean_abs={metrics['mean_abs']:.6f}  "
        f"max_abs={metrics['max_abs']:.6f}"
    )


@torch.no_grad()
def dump_smolvla_edge_fixture(
    ctx: ExportContext,
    *,
    prefix: dict,
    lm_hidden_states: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
) -> pathlib.Path:
    core = ctx.model
    noise = core.sample_noise(
        (prefix["inputs_embeds"].shape[0], core.config.chunk_size, core.config.max_action_dim),
        ctx.device,
    )
    timestep = torch.ones(prefix["inputs_embeds"].shape[0], device=ctx.device, dtype=torch.float32)
    from trt.diffusion import SmolVLAPrefixKVStepEncoder, StaticActionVelocityStep

    class _SmolVLAActionExpert(nn.Module):
        def __init__(self, model_core):
            super().__init__()
            self.vlm_with_expert = model_core.vlm_with_expert

        def forward(self, **kwargs):
            return self.vlm_with_expert.forward(**kwargs)

    action_module = StaticActionVelocityStep(
        step_encoder=SmolVLAPrefixKVStepEncoder(core),
        action_expert=_SmolVLAActionExpert(core),
        velocity_decoder=core.action_out_proj,
        output_tokens=int(core.config.chunk_size),
    ).eval().to(ctx.device)

    prefix_k = prefix_k.to(device=ctx.device, dtype=noise.dtype).contiguous()
    prefix_v = prefix_v.to(device=ctx.device, dtype=noise.dtype).contiguous()
    noise, timestep, prefix_k, prefix_v, position_ids, attention_mask = make_smolvla_runner_inputs(
        core,
        prefix["pad_mask"],
        prefix_k,
        prefix_v,
        noise,
        timestep,
        ctx.device,
    )
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
            prefix_pad_mask=prefix["pad_mask"],
        ),
        smolvla_action_adapter(core),
    )
    velocity_name = ctx.io.action.output_names[0]
    return dump_edge_fixture(
        str(ctx.engine_root),
        {
            "pixel_values": ctx.pixel_values.to(device=ctx.device).contiguous(),
            "inputs_embeds": prefix["inputs_embeds"].to(device=ctx.device, dtype=torch.float16),
            "prefix_k": prefix_k,
            "prefix_v": prefix_v,
            "lm_hidden_states": lm_hidden_states.to(device=ctx.device, dtype=torch.float16),
            "initial_actions": noise,
            "timestep": timestep,
            velocity_name: velocity,
            "actions_out": actions_out.to(device=ctx.device, dtype=noise.dtype),
        },
    )


class SmolVLAExportHooks(VLAExportHooks):
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
        images, img_masks, tokens, masks, state = prepare_smolvla_batch(
            ctx.policy,
            ctx.model_inputs,
            ctx.device,
        )
        ctx.pixel_values = images[0].contiguous()
        ctx.action_side = {
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "state": state,
            "batch_size": int(tokens.shape[0]),
        }
        ensure_smolvla_on_device(ctx.model, ctx.device)

    def build_vision_spec(self, ctx: ExportContext):
        return build_smolvla_vision_export_params(
            ctx.model,
            ctx.pixel_values,
            ctx.device,
            io=self.io,
            trt_settings=self.vision_trt_settings,
        )

    def pack_language_inputs(self, ctx: ExportContext) -> dict:
        return pack_smolvla_prefix(
            ctx.model,
            ctx.image_embs,
            ctx.action_side["img_masks"],
            ctx.action_side["tokens"],
            ctx.action_side["masks"],
            ctx.action_side["state"],
        )

    def build_language_spec(self, ctx: ExportContext):
        validate_language_len(ctx.language_inputs, ctx.max_seq_len)
        return build_smolvla_language_export_params(
            ctx.model,
            ctx.language_inputs,
            ctx.device,
            io=self.io,
            trt_settings=self.action_trt_settings,
        )

    def build_chat_template(self, tokenizer: Any) -> dict[str, Any]:
        """Minimal processed_chat_template.json for VitRunner image placeholder expansion."""
        image_format = tokenizer.decode([int(image_token_id)])
        if not image_format.strip():
            image_format = "<image>"
        return {
            "model_path": "smolvla-vitrunner",
            "roles": {
                "user": {"prefix": "", "suffix": ""},
            },
            "content_types": {
                "image": {"format": image_format},
            },
            "generation_prompt": "",
            "default_system_prompt": "",
        }

    def save_language_artifacts(self, ctx: ExportContext, language_dir: pathlib.Path) -> None:
        from trt.tokenizer import save_embedding_table, save_tokenizer_for_edge_llm

        self._last_image_token_id = int(ctx.lang_spec.image_token_id)
        save_embedding_table(ctx.lang_spec.language_model, language_dir)
        save_tokenizer_for_edge_llm(
            language_dir,
            tokenizer=self.tokenizer,
            chat_template=build_smolvla_vitrunner_chat_template(
                self.tokenizer,
                image_token_id=int(ctx.lang_spec.image_token_id),
            ),
        )

    def build_diffusion_spec(self, ctx: ExportContext):
        return build_smolvla_diffusion_export_params(
            ctx.model,
            prefix=ctx.language_inputs,
            device=ctx.device,
            io=self.io,
            trt_settings=self.action_trt_settings,
        )

    def after_export(self, ctx: ExportContext, sink: ExportSink) -> None:
        if not ctx.accuracy_check or not self.stage_parity:
            return
        if sink.mode is not ExportMode.SERIALIZED:
            return

        language_dir = ctx.engine_subdir("language")
        action_dir = ctx.engine_subdir("action")
        language_runner = SerializedPI05Language(SerializedTRTEngine(language_dir))
        text_cfg = smolvla_text_config(ctx.model)
        language_model = smolvla_language_model(ctx.model)
        max_seq_len = int(ctx.lang_spec.max_seq_len)
        num_layers = int(ctx.model.vlm_with_expert.num_vlm_layers)

        with torch.no_grad():
            trt_hidden, trt_prefix_k, trt_prefix_v = run_serialized_smolvla_language(
                language_runner,
                ctx.language_inputs["inputs_embeds"],
                max_seq_len=max_seq_len,
                num_layers=num_layers,
                device=ctx.device,
                position_ids=ctx.language_inputs.get("position_ids"),
                cfg=text_cfg,
                language_model=language_model,
            )

        action_runner = SerializedPI05Action(SerializedTRTEngine(action_dir))
        compare_smolvla_edge_pipeline_to_eager(
            core=ctx.model,
            policy=ctx.policy,
            images=ctx.action_side["images"],
            img_masks=ctx.action_side["img_masks"],
            tokens=ctx.action_side["tokens"],
            masks=ctx.action_side["masks"],
            state=ctx.action_side["state"],
            trt_image_embs=ctx.image_embs,
            trt_hidden=trt_hidden,
            trt_prefix_k=trt_prefix_k,
            trt_prefix_v=trt_prefix_v,
            action_runner=action_runner,
            device=ctx.device,
            seed=ctx.seed,
        )

        if ctx.engine_root is not None:
            fixture_dir = dump_smolvla_edge_fixture(
                ctx,
                prefix=ctx.language_inputs,
                lm_hidden_states=trt_hidden,
                prefix_k=trt_prefix_k,
                prefix_v=trt_prefix_v,
            )
            ctx.plugin_info["fixture_dir"] = str(fixture_dir)
            smoke_input = write_llm_runtime_smoke_case(
                ctx.engine_root,
                task_text=_smolvla_task_text(ctx.model_inputs),
                images=[_tensor_image_to_pil(ctx.pixel_values)],
                max_generate_length=self.max_generate_length,
            )
            ctx.plugin_info["runtime_smoke_input"] = str(smoke_input)

    def finalize_plugin_info(self, ctx: ExportContext) -> dict:
        info = super().finalize_plugin_info(ctx)
        pad_mask = ctx.language_inputs.get("pad_mask")
        if pad_mask is not None:
            info["prefix_seq_len"] = int(pad_mask.shape[1])
        info["chunk_size"] = int(ctx.model.config.chunk_size)
        info["max_action_dim"] = int(ctx.model.config.max_action_dim)
        info["output_action_dim"] = action_output_dim(ctx.policy)
        info["num_inference_steps"] = int(ctx.model.config.num_steps)
        if ctx.vis_spec is not None and ctx.vis_spec.config_seq_len:
            info["vision_output_seq_len"] = int(ctx.vis_spec.config_seq_len)
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
