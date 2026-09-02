"""EdgeExporter: Dynamo-style export() plus stock dynamo.compile.

Public API::

    exporter = EdgeExporter()
    compiled = exporter.export(model, inputs, config=EdgeConfig(...))
    compiled = exporter.export_for_policy(policy, inputs, config=EdgeConfig(...))

``export()`` is ``torch.export`` + ``torch_tensorrt.dynamo.compile`` (no fork).
Around ``torch.export`` it installs model wrappers (``apply_model_patches``)
then Edge-LLM plugin attention (``apply_edge_plugins``).
``export_for_policy`` looks up vision / language / fuse adapters, builds one
``PolicyStep``, and calls ``export()`` once so the partitioner emits the
screenshot hybrid graph. Model diffs live under ``models/<name>/``:
``adapters.py`` discovers towers + leftover fuse; ``patches/`` registers
``@register_model_patch`` wrappers. The public call does not change.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt
from torch.fx.node import Target

_GRAPH_DIR = Path(__file__).resolve().parent
_TEST_ROOT = _GRAPH_DIR.parent
for _path in (_TEST_ROOT, _GRAPH_DIR):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)


@torch.library.custom_op("edge::fuse_prefix", mutates_args=())
def fuse_prefix(vision_tokens: torch.Tensor, lang_embeds: torch.Tensor) -> torch.Tensor:
    return torch.cat([vision_tokens, lang_embeds], dim=1)


@fuse_prefix.register_fake  # type: ignore[misc]
def _(vision_tokens: torch.Tensor, lang_embeds: torch.Tensor) -> torch.Tensor:
    return torch.cat([vision_tokens, lang_embeds], dim=1)


FUSE_OP = torch.ops.edge.fuse_prefix.default


@torch.library.custom_op("edge::scatter_image_tokens", mutates_args=())
def scatter_image_tokens(
    vision_tokens: torch.Tensor,
    lang_embeds: torch.Tensor,
    image_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Splice flattened vision rows into image-token slots (GR00T / Eagle)."""
    hidden = lang_embeds.shape[-1]
    vis = vision_tokens.reshape(-1, hidden).to(dtype=lang_embeds.dtype)
    out = lang_embeds.clone()
    flat = out.reshape(-1, hidden)
    mask = image_token_mask.reshape(-1).to(dtype=torch.bool)
    n = int(mask.sum().item())
    if n:
        flat[mask] = vis[:n]
    return flat.reshape_as(lang_embeds)


@scatter_image_tokens.register_fake  # type: ignore[misc]
def _(
    vision_tokens: torch.Tensor,
    lang_embeds: torch.Tensor,
    image_token_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(lang_embeds)


SCATTER_OP = torch.ops.edge.scatter_image_tokens.default


@dataclass(frozen=True)
class FuseSpec:
    """Leftover glue op for PolicyStep. extra_keys are tensors after lang_embeds."""

    op: Target
    extra_keys: tuple[str, ...] = ()


@dataclass
class CompareReport:
    """Eager SDPA (A) vs compiled TRT (C) for one hybrid graph."""

    parity: dict[str, dict[str, float]]
    eager_ms: float
    compiled_ms: float
    speedup: float


@dataclass
class EdgeConfig:
    """Dynamo export knobs plus compile kwargs forwarded to dynamo.compile."""

    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False

    min_block_size: int = 1
    require_full_compilation: bool = False
    torch_executed_ops: set[Target] = field(default_factory=set)
    decompose_attention: bool = False
    immutable_weights: bool = True
    extra_compile_kwargs: dict[str, Any] = field(default_factory=dict)

    compare: bool = False
    compare_warmup: int = 5
    compare_iters: int = 100

    def compile_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "min_block_size": self.min_block_size,
            "require_full_compilation": self.require_full_compilation,
            "torch_executed_ops": set(self.torch_executed_ops),
            "decompose_attention": self.decompose_attention,
            "immutable_weights": self.immutable_weights,
        }
        kwargs.update(self.extra_compile_kwargs)
        return kwargs


def _as_args_kwargs(inputs: Any) -> tuple[tuple, dict]:
    if isinstance(inputs, Mapping):
        return (), dict(inputs)
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        return tuple(inputs), {}
    return (inputs,), {}


def _is_dryrun(config: EdgeConfig) -> bool:
    return bool(config.extra_compile_kwargs.get("dryrun"))


def _call(model: nn.Module, args: tuple, kwargs: dict):
    if args and kwargs:
        return model(*args, **kwargs)
    if kwargs:
        return model(**kwargs)
    return model(*args)


def _as_tuple(out) -> tuple:
    return out if isinstance(out, tuple) else (out,)


def _output_names(n: int) -> tuple[str, ...]:
    named = ("logits", "hidden", "prefix_k", "prefix_v")
    if n == len(named):
        return named
    return tuple(f"output{i}" for i in range(n))


def _device_of(args: tuple, kwargs: dict) -> torch.device:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cuda")


def _cuda_bench(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    """Same timing loop as ``test_vla_pi05_e2e``: warmup, then CUDA events / iters."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type != "cuda":
            import time

            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) * 1000.0 / max(iters, 1)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / max(iters, 1)


_ADAPTERS: dict[str, list[tuple[Callable[[nn.Module], bool], Callable]]] = {
    "vision": [],
    "language": [],
    "fuse": [],
}

_PATCHES: list[tuple[Callable[[nn.Module], bool], Callable]] = []


def register_edge_adapter(component: str, *, match: Callable[[nn.Module], bool]):
    """HF-style registry: first matching adapter wins. Import-time registration."""
    if component not in _ADAPTERS:
        raise ValueError(f"unknown adapter component {component!r}; expected {tuple(_ADAPTERS)}")

    def decorator(fn: Callable) -> Callable:
        _ADAPTERS[component].append((match, fn))
        return fn

    return decorator


def register_model_patch(*, match: Callable[[nn.Module], bool]):
    """HF-style wrap factory: first match wins. Runs around ``torch.export``."""

    def decorator(factory: Callable) -> Callable:
        _PATCHES.append((match, factory))
        return factory

    return decorator


def source_policy(module: nn.Module) -> nn.Module:
    """Policy ``export_for_policy`` stashed on PolicyStep; otherwise ``module``."""
    src = getattr(module, "_source", None)
    if isinstance(src, tuple) and src:
        return src[0]
    return module


def _patch_inputs(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        out: dict[str, Any] = {}
        if inputs:
            out["pixel_values"] = inputs[0]
        if len(inputs) > 1:
            out["lang_embeds"] = inputs[1]
        return out
    return {"pixel_values": inputs}


@contextmanager
def apply_tower_wrappers(
    module: nn.Module,
    inputs: Any,
    *,
    wrap_vision: Callable,
    wrap_language: Callable,
):
    """Temporarily replace PolicyStep vision/language with ExportModules."""
    source = source_policy(module)
    patch_inputs = _patch_inputs(inputs)
    originals: dict[str, nn.Module] = {}
    try:
        if hasattr(module, "vision_tower"):
            originals["vision_tower"] = module.vision_tower
            module.vision_tower = wrap_vision(source, patch_inputs)
        if hasattr(module, "language_model"):
            originals["language_model"] = module.language_model
            module.language_model = wrap_language(source, patch_inputs)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


_MODEL_ADAPTERS_LOADED = False


def _ensure_model_adapters() -> None:
    """Import ``models.*`` so ``@register_edge_adapter`` rows attach."""
    global _MODEL_ADAPTERS_LOADED
    if _MODEL_ADAPTERS_LOADED:
        return
    import models  # noqa: F401

    _MODEL_ADAPTERS_LOADED = True


def _lookup(component: str, policy: nn.Module, inputs: Mapping[str, Any]):
    _ensure_model_adapters()
    for match, fn in _ADAPTERS[component]:
        if match(policy):
            return fn(policy, inputs)
    return None


def _as_fuse_spec(value: Any) -> FuseSpec:
    if value is None:
        return FuseSpec(FUSE_OP)
    if isinstance(value, FuseSpec):
        op = value.op.default if hasattr(value.op, "default") else value.op
        return FuseSpec(op, value.extra_keys)
    op = value.default if hasattr(value, "default") else value
    return FuseSpec(op)


def _adapt_vision(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    adapted = _lookup("vision", policy, inputs)
    if adapted is not None:
        return adapted
    vision = getattr(policy, "vision_tower", None)
    if vision is None:
        raise AttributeError("policy has no vision_tower; cannot adapt vision for Edge export")
    return vision.eval()


def _adapt_language(policy: nn.Module, inputs: Mapping[str, Any]) -> nn.Module:
    adapted = _lookup("language", policy, inputs)
    if adapted is not None:
        return adapted
    language = getattr(policy, "language_model", None)
    if language is None:
        raise AttributeError(
            "policy has no language_model; cannot adapt language for Edge export"
        )
    return language.eval()


class PolicyStep(nn.Module):
    """Screenshot topology: vision engine | leftover fuse op | language engine."""

    def __init__(
        self,
        vision: nn.Module,
        language: nn.Module,
        *,
        fuse: Target | None = None,
        fuse_extra_keys: tuple[str, ...] = (),
        policy: nn.Module | None = None,
    ):
        super().__init__()
        self.vision_tower = vision
        self.language_model = language
        self.fuse_op = fuse or FUSE_OP
        self.fuse_extra_keys = fuse_extra_keys
        # Tuple so the source policy is not registered as a submodule.
        self._source = (policy,) if policy is not None else ()

    def forward(
        self,
        pixel_values: torch.Tensor,
        lang_embeds: torch.Tensor,
        *args: torch.Tensor,
    ):
        n_extra = len(self.fuse_extra_keys)
        extra, lm_args = args[:n_extra], args[n_extra:]
        vision_tokens = self.vision_tower(pixel_values)
        if vision_tokens.ndim == 2:
            batch = lang_embeds.shape[0]
            vision_tokens = vision_tokens.reshape(batch, -1, lang_embeds.shape[-1])
        prefix = self.fuse_op(vision_tokens, lang_embeds, *extra)
        return self.language_model(prefix, *lm_args)


def _vision_transformer(module: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    """Return (encoder host with .encoder.layers, wrapper used for seq/batch)."""
    candidates = [
        module,
        getattr(module, "vision_tower", None),
        getattr(module, "vision_model", None),
    ]
    for cand in candidates:
        if cand is None:
            continue
        inner = cand
        for _ in range(3):
            if hasattr(inner, "encoder"):
                return inner, cand
            nxt = getattr(inner, "vision_model", None)
            if nxt is None or nxt is inner:
                break
            inner = nxt
    return None, None


def _language_lm(module: nn.Module) -> nn.Module | None:
    candidates = [module, getattr(module, "language_model", None)]
    for cand in candidates:
        if cand is None:
            continue
        lm = getattr(cand, "lm", cand)
        cfg = getattr(lm, "config", None)
        if cfg is not None and hasattr(lm, "layers"):
            return lm
    return None


@contextmanager
def apply_model_patches(module: nn.Module, inputs: Any = None):
    """Install the first matching ``@register_model_patch`` wrap for the trace."""
    _ensure_model_adapters()
    factory = next((fn for match, fn in _PATCHES if match(module)), None)
    if factory is None:
        yield
        return
    with factory(module, inputs):
        yield


@contextmanager
def apply_edge_plugins(module: nn.Module):
    """Reversible Edge-LLM plugin attention swaps (after wrappers are installed)."""
    patched_groups: list = []
    try:
        vision, vision_wrap = _vision_transformer(module)
        if vision is not None:
            from trt.plugin.plugin_utils import patch_vision_attention

            patched_groups.append(
                patch_vision_attention(
                    vision,
                    batch_size=int(getattr(vision_wrap, "batch_size", 1) or 1),
                    seq_len=int(getattr(vision_wrap, "seq_len", 1) or 1),
                    name="vision",
                )
            )

        lm = _language_lm(module)
        cfg = getattr(lm, "config", None) if lm is not None else None
        if lm is not None and cfg is not None:
            from trt.plugin.attention import ContextAttentionMaskType
            from trt.plugin.plugin_utils import patch_language_attention

            head_dim = int(
                getattr(cfg, "head_dim", 0)
                or cfg.hidden_size // cfg.num_attention_heads
            )
            patched_groups.append(
                patch_language_attention(
                    lm,
                    hidden_size=int(cfg.hidden_size),
                    num_attention_heads=int(cfg.num_attention_heads),
                    num_key_value_heads=int(cfg.num_key_value_heads),
                    head_dim=head_dim,
                    context_attention_mask_type=ContextAttentionMaskType.PADDING,
                )
            )
        yield
    finally:
        if patched_groups:
            from trt.plugin.plugin_utils import restore_attention

            for group in patched_groups:
                restore_attention(group)


class EdgeExporter:
    def __init__(self) -> None:
        self.last_compare: CompareReport | None = None

    def export(
        self,
        model: nn.Module,
        inputs: Any,
        config: EdgeConfig | dict[str, Any],
    ) -> torch.fx.GraphModule:
        if isinstance(config, dict):
            config = EdgeConfig(**config)
        elif not isinstance(config, EdgeConfig):
            raise TypeError(f"Expected EdgeConfig or dict, got {type(config)}")

        args, kwargs = _as_args_kwargs(inputs)
        compare = bool(config.compare) and not _is_dryrun(config)
        if config.compare and _is_dryrun(config):
            print("compare=True ignored during dryrun (no engines)")

        eager_out = None
        eager_ms: float | None = None
        with apply_model_patches(model, inputs):
            if compare:
                model.eval()
                with torch.no_grad():
                    eager_out = _call(model, args, kwargs)
                device = _device_of(args, kwargs)
                eager_ms = _cuda_bench(
                    lambda: _call(model, args, kwargs),
                    warmup=config.compare_warmup,
                    iters=config.compare_iters,
                    device=device,
                )

            with apply_edge_plugins(model):
                exported = torch.export.export(
                    model.eval(),
                    args=args,
                    kwargs=kwargs or None,
                    strict=config.strict,
                    dynamic_shapes=config.dynamic_shapes,
                    prefer_deferred_runtime_asserts_over_guards=(
                        config.prefer_deferred_runtime_asserts_over_guards
                    ),
                )

        compile_kwargs = config.compile_kwargs()
        if args:
            compile_kwargs["arg_inputs"] = args
        if kwargs:
            compile_kwargs["kwarg_inputs"] = kwargs
        if not args and not kwargs:
            raise ValueError("export() requires positional or keyword inputs")
        compiled = torch_tensorrt.dynamo.compile(exported, **compile_kwargs)

        if compare:
            assert eager_out is not None and eager_ms is not None
            self.last_compare = self._compare_eager(
                compiled, args, kwargs, eager_out, eager_ms, config
            )
        else:
            self.last_compare = None
        return compiled

    def _compare_eager(
        self,
        compiled: nn.Module,
        args: tuple,
        kwargs: dict,
        eager_out,
        eager_ms: float,
        config: EdgeConfig,
    ) -> CompareReport:
        from trt.measure import parity

        device = _device_of(args, kwargs)
        with torch.no_grad():
            compiled_out = _call(compiled, args, kwargs)

        eager_t, trt_t = _as_tuple(eager_out), _as_tuple(compiled_out)
        names = _output_names(min(len(eager_t), len(trt_t)))
        report: dict[str, dict[str, float]] = {}
        for name, a, b in zip(names, eager_t, trt_t):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                continue
            report[name] = parity(f"A vs C ({name})", a, b)

        compiled_ms = _cuda_bench(
            lambda: _call(compiled, args, kwargs),
            warmup=config.compare_warmup,
            iters=config.compare_iters,
            device=device,
        )
        speedup = eager_ms / max(compiled_ms, 1e-9)
        print(f"eager execute: {eager_ms:.3f} ms")
        print(f"trt execute:   {compiled_ms:.3f} ms")
        print(f"speedup:       {speedup:.2f}x")
        return CompareReport(report, eager_ms, compiled_ms, speedup)

    def export_for_policy(
        self,
        policy: nn.Module,
        inputs: Mapping[str, Any],
        config: EdgeConfig | dict[str, Any],
    ) -> torch.fx.GraphModule:
        """Compose one PolicyStep and compile it — screenshot graph, not a dict of engines."""
        if isinstance(config, dict):
            config = EdgeConfig(**config)

        fuse = _as_fuse_spec(_lookup("fuse", policy, inputs))
        config = replace(
            config,
            torch_executed_ops=set(config.torch_executed_ops) | {fuse.op},
        )

        step = PolicyStep(
            _adapt_vision(policy, inputs),
            _adapt_language(policy, inputs),
            fuse=fuse.op,
            fuse_extra_keys=fuse.extra_keys,
            policy=policy,
        ).eval()
        return self.export(
            step, _policy_step_inputs(inputs, fuse.extra_keys), config=config
        )


_LM_ARG_KEYS = (
    "rope_rotary_cos_sin",
    "context_lengths",
    "kvcache_start_index",
    "last_token_ids",
    "ds_stack",
)


def _policy_step_inputs(
    inputs: Mapping[str, Any],
    fuse_extra_keys: tuple[str, ...] = (),
) -> tuple:
    args = [inputs["pixel_values"], inputs["lang_embeds"]]
    for key in fuse_extra_keys:
        if key not in inputs:
            raise KeyError(f"fuse extra input {key!r} missing from PolicyStep inputs")
        args.append(inputs[key])
    for key in _LM_ARG_KEYS:
        if key in inputs:
            args.append(inputs[key])
    args.extend(inputs.get("past_key_values") or ())
    return tuple(args)


def _inline_torch_modules_by_args(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Map GPU-island placeholders to call_module args by position.

    Torch-TRT's inliner matches leftover-submodule placeholders by *name*. That
    breaks the screenshot graph: ``fuse_prefix`` takes a TRT output *and* a parent
    placeholder (``lang_embeds``), so only some names collide and the TRT output
    is left as a stray placeholder (``permute`` / ``clone``).
    """
    from torch_tensorrt.dynamo._exporter import copy_submodule_attributes

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    for gm_node in list(gm.graph.nodes):
        if gm_node.op != "call_module" or "_run_on_gpu" not in gm_node.name:
            continue
        submodule = getattr(gm, gm_node.name)
        placeholders = [n for n in submodule.graph.nodes if n.op == "placeholder"]
        submodule_inputs = gm_node.args
        if len(placeholders) != len(submodule_inputs):
            raise RuntimeError(
                f"{gm_node.name} has {len(placeholders)} placeholders "
                f"but {len(submodule_inputs)} args"
            )
        with gm.graph.inserting_before(gm_node):
            val_map = dict(zip(placeholders, submodule_inputs))
            submodule_output = gm.graph.graph_copy(submodule.graph, val_map)
            if isinstance(submodule_output, tuple):
                getitem_users = [
                    user
                    for user in list(gm_node.users.keys())
                    if user.op == "call_function" and user.target is operator.getitem
                ]
                for user in getitem_users:
                    _, idx = user.args
                    user.replace_all_uses_with(submodule_output[idx])
                    gm.graph.erase_node(user)
            else:
                gm_node.replace_all_uses_with(submodule_output)
            copy_submodule_attributes(gm, submodule, gm_node.name)
        gm.graph.erase_node(gm_node)
    return gm


def _patch_torch_tensorrt_gpu_inliner() -> None:
    from torch_tensorrt.dynamo import _exporter as trt_exporter

    if getattr(trt_exporter.inline_torch_modules, "_edge_by_args", False):
        return
    _inline_torch_modules_by_args._edge_by_args = True  # type: ignore[attr-defined]
    trt_exporter.inline_torch_modules = _inline_torch_modules_by_args


_patch_torch_tensorrt_gpu_inliner()


def dump_graph(gm: torch.fx.GraphModule | torch.fx.Graph, title: str) -> None:
    graph = gm.graph if isinstance(gm, torch.fx.GraphModule) else gm
    print(f"\n===== {title} =====")
    for node in graph.nodes:
        if node.op == "placeholder":
            print(f"  %{node.name} = placeholder")
        elif node.op == "get_attr":
            print(f"  %{node.name} = prim::get_attr[{node.target}]")
        elif node.op == "call_module":
            print(f"  %{node.name} = call_module[{node.target}]{tuple(node.args)}")
        elif node.op == "call_function":
            print(f"  %{node.name} = {node.target}{tuple(node.args)}")
        elif node.op == "output":
            print(f"  return {node.args[0]}")
        else:
            print(f"  %{node.name} = {node.op} {node.target} {node.args}")


def named_runtime_preview(gm: torch.fx.GraphModule | torch.fx.Graph) -> None:
    """Graph 3: rename only. First execute_engine → vision_tower, second → text_encoder."""
    graph = gm.graph if isinstance(gm, torch.fx.GraphModule) else gm
    engines = [
        n
        for n in graph.nodes
        if n.op == "call_function" and "execute_engine" in str(n.target)
    ]
    aliases = ["tensorrt::vision_tower", "tensorrt::text_encoder"]
    print("\n===== graph 3 (rename) =====")
    for node, alias in zip(engines, aliases):
        print(f"  %{node.name}  {node.target}  →  {alias}")
    if len(engines) < 2:
        print("  (expected two execute_engine nodes; partitioner cut differently)")
