"""PI0.5 staged export + inference through torch-bear / GraphIR (no TensorRT).

Same split as ``vla/test_vla_pi05_e2e.py``:

  vision    -> GridVisionExportModule
  language  -> HF decoder, last_hidden_state (prefix K/V is a separate eager pass)
  diffusion -> one action-velocity step, reused by the host Euler loop

Each stage is ``torch.export`` + ``compile_fx(..., emit="graphir-mlir")``.
GraphIR compiles those three tensor graphs. Host Python still:

  1. stitches vision tokens + language embeddings into the prefix
  2. takes prefix K/V from eager ``use_cache=True`` (not a GraphIR graph)
  3. runs Euler from t=1 (noise) to t=0 (actions), calling diffusion each step

There are no TensorRT engines and no TRT attention plugins. Unsupported ATen
ops fall back to PyTorch.

Run::

    python mlir/test_vla_pi05_graphir_e2e.py
"""

from __future__ import annotations

import logging
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

_TEST_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = _TEST_ROOT.parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

_GRAPHIR_ROOT = _WORKSPACE / "GraphIR"
_GRAPHIR_PYTHON = _GRAPHIR_ROOT / "python"
_TORCH_BEAR_ROOT = _GRAPHIR_ROOT / "torch-bear"
for _p in (_GRAPHIR_PYTHON, _TORCH_BEAR_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from trt.data import frame_from_test_data, load_test_data
from trt.measure import parity
from trt.modules.export.diffusion import (
    PI05PrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)
from trt.modules.export.vision import GridVisionExportModule
from trt.utils import (
    configure_thor_pytorch,
    force_hf_attention,
    free_cuda_memory,
    move_pi05_diffusion_modules_to_device,
)

configure_thor_pytorch()

logger = logging.getLogger("pi05_graphir")

_WARMUP = 5
_ITERS = 20
_DENOISE_STEPS = 10
_ATEN_SHIM_FALLBACK = "allow_python"


def _import_compile_fx():
    try:
        import graphir  # noqa: F401  — C++ JIT (GraphIR/python)
        import torch_bear  # noqa: F401  — registers the torch-bear backend
        from torch_bear.frontends import compile_fx
    except ImportError as exc:
        raise SystemExit(
            "torch-bear/GraphIR is not importable. Build GraphIR from source, then:\n"
            f"  export PYTHONPATH={_GRAPHIR_PYTHON}:{_TORCH_BEAR_ROOT}:$PYTHONPATH\n"
            f"Original error: {exc}"
        ) from exc
    return compile_fx


def _cuda_ms(fn: Callable[[], Any], device: torch.device, *, warmup: int, iters: int) -> float:
    if device.type != "cuda":
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / max(iters, 1)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / max(iters, 1)


def _as_tuple(out: Any) -> tuple:
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def _speedup(eager_ms: float, compiled_ms: float) -> str:
    if eager_ms <= 0.0 or compiled_ms <= 0.0:
        return "n/a"
    return f"{eager_ms / compiled_ms:.3f}x"


def euler_integrate_velocity(
    velocity_fn: Callable,
    noise: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """PI05 / openpi Euler: t=1 (noise) -> t=0 (actions), ``x <- x + dt * v``."""
    dt = -1.0 / num_steps
    x_t = noise
    bsz = noise.shape[0]
    device = noise.device
    for step in range(num_steps):
        time = 1.0 + step * dt
        timestep = torch.full((bsz,), time, device=device, dtype=torch.float32)
        velocity = _as_tuple(velocity_fn(x_t, timestep))[0]
        x_t = x_t + dt * velocity
    return x_t


def _stack_prefix_kv(past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten HF / PrefixKV cache objects into stacked [L, B, H, S, D] tensors."""
    if past_key_values is None:
        raise ValueError("language forward returned no past_key_values")

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache, past_key_values.value_cache

    if hasattr(past_key_values, "layers"):
        layers = list(past_key_values.layers)
        keys = [getattr(layer, "keys", None) for layer in layers]
        values = [getattr(layer, "values", None) for layer in layers]
        if any(k is None or v is None for k, v in zip(keys, values)):
            raise ValueError("Cache layers are missing keys/values")
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            return (
                torch.stack([kv[0] for kv in past_key_values], dim=0),
                torch.stack([kv[1] for kv in past_key_values], dim=0),
            )
        if len(past_key_values) == 2 and torch.is_tensor(past_key_values[0]):
            return past_key_values[0], past_key_values[1]

    raise TypeError(f"unsupported past_key_values type: {type(past_key_values)!r}")


class PI05LanguageGraphIRModule(nn.Module):
    """PaliGemma decoder for GraphIR: embeddings in, last hidden out.

    ``use_cache=False`` so export stays tensor-only. HuggingFace DynamicCache
    is not a good GraphIR graph (same reason TRT used a custom KV plugin).
    Prefix K/V for diffusion is taken from a separate eager cache pass.
    """

    def __init__(self, language: nn.Module):
        super().__init__()
        self.language = language
        self._lm_dtype = next(language.parameters()).dtype

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        out = self.language(
            inputs_embeds=inputs_embeds.to(dtype=self._lm_dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def eager_prefix_kv(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.language(
            inputs_embeds=inputs_embeds.to(dtype=self._lm_dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        return _stack_prefix_kv(out.past_key_values)


def _call_lifted_graphir(compiled: Callable, exported, user_args: tuple) -> Any:
    """Bind export-lifted weights so the GraphIR callable can be used like the module."""
    from torch.export.graph_signature import InputKind

    state = {**exported.state_dict, **exported.constants}
    device = next((a.device for a in user_args if torch.is_tensor(a)), None)
    user = iter(user_args)
    full: list[Any] = []
    for spec in exported.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            full.append(next(user))
            continue
        value = state[spec.target]
        if torch.is_tensor(value) and device is not None and value.device != device:
            value = value.to(device=device)
        full.append(value)
    return compiled(*full)


def compile_module_graphir(
    module: nn.Module,
    args: tuple,
    *,
    label: str,
    compile_fx: Callable,
    aten_shim_fallback: str = _ATEN_SHIM_FALLBACK,
    dump_dir: Path | None = None,
) -> tuple[Callable, float, float]:
    """Export an nn.Module and compile the FX graph through GraphIR."""
    module = module.eval()
    t0 = time.perf_counter()
    exported = torch.export.export(module, args=args, strict=False)
    export_s = time.perf_counter() - t0
    # Use the lifted graph (weights are extra args). GraphIR's C++ parser does
    # not accept leftover tb.call_module from exported.module().
    gm = exported.graph_module
    get_attrs = [n.target for n in gm.graph.nodes if n.op == "get_attr"]
    if get_attrs:
        print(f"[{label} graphir] export get_attr targets ({len(get_attrs)}): {get_attrs[:12]}")

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        from torch_bear.frontends import _fx_graph_to_func_op
        from xdsl.printer import Printer

        func_op = _fx_graph_to_func_op(gm)
        buf = StringIO()
        Printer(stream=buf).print_op(func_op)
        out_path = dump_dir / f"{label}_aten.mlir"
        out_path.write_text(buf.getvalue())
        print(f"[{label} graphir] wrote ATen MLIR to {out_path}")

    t1 = time.perf_counter()
    compiled = compile_fx(
        gm,
        list(args),
        emit="graphir-mlir",
        aten_shim_fallback=aten_shim_fallback,
    )
    compile_s = time.perf_counter() - t1
    print(
        f"[{label} graphir] export={export_s:.3f}s  "
        f"compile={compile_s:.3f}s  total={export_s + compile_s:.3f}s"
    )

    def _fn(*user_args):
        return _call_lifted_graphir(compiled, exported, user_args)

    return _fn, export_s, compile_s


def build_pi05_prefix_embs(pi05_model, img_masks, tokens, masks, image_embs_list):
    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []

    for img_emb, img_mask in zip(image_embs_list, img_masks, strict=True):
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

    lang_emb = pi05_model.paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)
    pad_masks.append(masks)

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    valid = prefix_pad_masks.to(device=prefix_embs.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError("build_pi05_prefix_embs requires equal valid token counts across the batch")

    compact_len = int(valid_counts[0].item())
    compact_embs = torch.stack(
        [prefix_embs[b, valid[b], :] for b in range(prefix_embs.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [prefix_position_ids[b, valid[b]] for b in range(prefix_embs.shape[0])],
        dim=0,
    )
    compact_pad_mask = torch.ones(
        prefix_embs.shape[0],
        compact_len,
        device=prefix_pad_masks.device,
        dtype=torch.bool,
    )
    compact_attention_mask = torch.zeros(
        prefix_embs.shape[0],
        1,
        compact_len,
        compact_len,
        device=prefix_embs.device,
        dtype=torch.float32,
    )
    return compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids


def make_pi05_suffix_position_and_mask(core, prefix_pad_masks, x_t, device):
    batch_size, suffix_len = x_t.shape[:2]
    prefix_pad_masks = prefix_pad_masks.to(device=device)
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + [0] * (suffix_len - 1),
        dtype=torch.int64,
        device=device,
    )[None, :].expand(batch_size, -1)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    del core
    attention_mask = prepare_attention_masks_4d(full_att_2d_masks)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask


def load_config():
    config = PI05Config(
        device="cpu",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=32,
        max_action_dim=32,
        image_resolution=(224, 224),
        input_features={
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    )
    config.validate_features()
    policy = PI05Policy(config).eval()
    return config, policy


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    compile_fx = _import_compile_fx()

    if not torch.cuda.is_available():
        raise SystemExit("GraphIR JIT needs CUDA.")

    device = torch.device("cuda")
    dtype = torch.float16

    config, policy = load_config()
    model = policy.model.to(device=device, dtype=dtype).eval()
    paligemma = model.paligemma_with_expert.paligemma.model
    vision = paligemma.vision_tower
    language = paligemma.language_model

    force_hf_attention(vision, "eager")
    force_hf_attention(language, "eager")

    pre_processor, _post_processor = make_pre_post_processors(
        config,
        None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    data = load_test_data("lerobot/libero", episode_index=0, frame_index=0)
    frame = frame_from_test_data(data, policy, fill_missing=True)
    model_inputs = pre_processor(frame)

    images, img_masks = policy._preprocess_images(model_inputs)
    tokens = model_inputs[OBS_LANGUAGE_TOKENS].to(device=device, dtype=torch.long)
    masks = model_inputs[OBS_LANGUAGE_ATTENTION_MASK].to(device=device, dtype=torch.bool)

    pixel_values = torch.cat(
        [img.to(device=device, dtype=dtype) for img in images],
        dim=0,
    ).contiguous()
    
    projector = paligemma.multi_modal_projector
    vision = vision.float()

    visual = GridVisionExportModule(
        vision_model=vision,
        projector=projector,
        sample_pixel_values=pixel_values.float(),
        select_layer=-1,
        pixel_shuffle=False,
        downsample_ratio=0.5,
        force_float32_input=True,
        vision_kwargs={},
    ).eval().to(device=device)

    with torch.no_grad():
        embs_eager = visual(pixel_values)
    vision_eager_ms = _cuda_ms(lambda: visual(pixel_values), device, warmup=_WARMUP, iters=_ITERS)

    vision_graphir, _, _ = compile_module_graphir(
        visual,
        (pixel_values,),
        label="vision",
        compile_fx=compile_fx,
    )
    with torch.no_grad():
        embs_graphir = vision_graphir(pixel_values)
    embs_graphir = _as_tuple(embs_graphir)[0]
    parity("PI05 vision eager vs GraphIR", embs_eager, embs_graphir)
    vision_graphir_ms = _cuda_ms(
        lambda: vision_graphir(pixel_values),
        device,
        warmup=_WARMUP,
        iters=_ITERS,
    )
    vision_tokens = embs_graphir

    per_camera_batch = int(images[0].shape[0])
    image_embs_list = list(
        vision_tokens.reshape(len(images), per_camera_batch, -1, vision_tokens.shape[-1])
    )
    inputs_embeds, prefix_pad_mask, prefix_attention_mask, prefix_position_ids = build_pi05_prefix_embs(
        model,
        img_masks,
        tokens,
        masks,
        image_embs_list,
    )
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()

    free_cuda_memory(visual, embs_eager, vision_graphir)
    vision.cpu()
    paligemma.multi_modal_projector.cpu()
    model.paligemma_with_expert.gemma_expert.cpu()
    free_cuda_memory()

    lang_module = PI05LanguageGraphIRModule(language).eval().to(device=device)
    lang_args = (inputs_embeds, prefix_attention_mask, prefix_position_ids)

    with torch.no_grad():
        lm_hidden_eager = lang_module(*lang_args)
    language_eager_ms = _cuda_ms(lambda: lang_module(*lang_args), device, warmup=_WARMUP, iters=_ITERS)

    language_graphir, _, _ = compile_module_graphir(
        lang_module,
        lang_args,
        label="language",
        compile_fx=compile_fx,
    )
    with torch.no_grad():
        lm_hidden_g = _as_tuple(language_graphir(*lang_args))[0]
    parity("PI05 language eager vs GraphIR", lm_hidden_eager, lm_hidden_g)
    language_graphir_ms = _cuda_ms(
        lambda: language_graphir(*lang_args),
        device,
        warmup=_WARMUP,
        iters=_ITERS,
    )

    with torch.no_grad():
        prefix_k, prefix_v = lang_module.eager_prefix_kv(*lang_args)
    prefix_k = prefix_k.to(device=device, dtype=dtype).contiguous()
    prefix_v = prefix_v.to(device=device, dtype=dtype).contiguous()

    free_cuda_memory(language_graphir, lang_module, lm_hidden_eager, language)
    model.cpu()
    free_cuda_memory()
    move_pi05_diffusion_modules_to_device(model, device, dtype)
    force_hf_attention(model.paligemma_with_expert.gemma_expert.model, "eager")

    bsz = inputs_embeds.shape[0]
    diffusion_model = StaticActionVelocityStepExportModule(
        step_encoder=PI05PrefixKVStepEncoderExportModule(model),
        action_expert=model.paligemma_with_expert.gemma_expert.model,
        velocity_decoder=model.action_out_proj,
        output_tokens=model.config.chunk_size,
        cast_hidden_fp32=False,
    ).eval().to(device=device)

    step_actions = torch.randn(
        bsz,
        model.config.chunk_size,
        model.config.max_action_dim,
        device=device,
        dtype=dtype,
    )
    step_timestep = torch.full((bsz,), 1.0, device=device, dtype=torch.float32)
    suffix_position_ids, suffix_attention_mask = make_pi05_suffix_position_and_mask(
        model,
        prefix_pad_mask,
        step_actions,
        device,
    )
    diffusion_input = (
        step_actions,
        step_timestep,
        prefix_k,
        prefix_v,
        suffix_position_ids,
        suffix_attention_mask,
    )

    with torch.no_grad():
        eager_velocity = diffusion_model(*diffusion_input)
    diffusion_eager_ms = _cuda_ms(
        lambda: diffusion_model(*diffusion_input),
        device,
        warmup=_WARMUP,
        iters=_ITERS,
    )

    diffusion_graphir, _, _ = compile_module_graphir(
        diffusion_model,
        diffusion_input,
        label="diffusion",
        compile_fx=compile_fx,
    )
    with torch.no_grad():
        graphir_velocity = _as_tuple(diffusion_graphir(*diffusion_input))[0]
    parity("PI05 diffusion eager vs GraphIR", eager_velocity, graphir_velocity)
    diffusion_graphir_ms = _cuda_ms(
        lambda: diffusion_graphir(*diffusion_input),
        device,
        warmup=_WARMUP,
        iters=_ITERS,
    )

    noise = torch.randn_like(step_actions)

    def _one_step(x_t: torch.Tensor, timestep: torch.Tensor):
        return diffusion_graphir(
            x_t,
            timestep,
            prefix_k,
            prefix_v,
            suffix_position_ids,
            suffix_attention_mask,
        )

    with torch.no_grad():
        actions = euler_integrate_velocity(_one_step, noise, _DENOISE_STEPS)
    print(
        f"[diffusion graphir] Euler {_DENOISE_STEPS} steps  "
        f"actions={tuple(actions.shape)}  dtype={actions.dtype}"
    )

    eager_total_ms = vision_eager_ms + language_eager_ms + diffusion_eager_ms
    graphir_total_ms = vision_graphir_ms + language_graphir_ms + diffusion_graphir_ms

    print()
    print("=== GraphIR stage execute (no TensorRT) ===")
    print(f"  vision     eager={vision_eager_ms:.3f} ms  graphir={vision_graphir_ms:.3f} ms  speedup={_speedup(vision_eager_ms, vision_graphir_ms)}")
    print(f"  language   eager={language_eager_ms:.3f} ms  graphir={language_graphir_ms:.3f} ms  speedup={_speedup(language_eager_ms, language_graphir_ms)}")
    print(f"  diffusion  eager={diffusion_eager_ms:.3f} ms  graphir={diffusion_graphir_ms:.3f} ms  speedup={_speedup(diffusion_eager_ms, diffusion_graphir_ms)}")
    print(f"  total      eager={eager_total_ms:.3f} ms  graphir={graphir_total_ms:.3f} ms  speedup={_speedup(eager_total_ms, graphir_total_ms)}")
    print(f"  actions    {tuple(actions.shape)}  (host Euler over GraphIR velocity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
