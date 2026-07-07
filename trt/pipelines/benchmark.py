from __future__ import annotations

from trt.context import BenchmarkResult, EdgeContext
from trt.config.execution_mode import ExecutionMode
from trt.config.parity_mode import ParityMode
from trt.config.pipeline_registry import get_inference_pipeline
from trt.executor.models.groot.inference.pipeline import STAGE_PARITY_TENSORS
from trt.measure import parity_metrics
from trt.pipelines.inference import InferencePipeline

SEED = 42

_SERIALIZED_ENGINE_DIRS = ("visual", "language", "action_context", "action")


def _print_parity_table(reference: str, backend: str, rows: list[tuple[str, str, dict[str, float]]]) -> None:
    header = (
        f"{'Stage':<16} {'Tensor':<14} "
        f"{'Mean Abs':>10} {'Max Abs':>10} {'Rel L2':>8} {'Rel Mean':>9} {'Close':>7}"
    )
    rule = "-" * len(header)

    print()
    print(f"Parity: {reference} vs {backend}")
    print(rule)
    print(header)
    print(rule)

    for stage_name, tensor_name, metrics in rows:
        print(
            f"{stage_name:<16} {tensor_name:<14} "
            f"{metrics['mean_abs']:>10.6f} {metrics['max_abs']:>10.6f} "
            f"{metrics['rel_l2']:>8.4f} {metrics['rel_mean_pct']:>8.2f}% "
            f"{metrics['close_pct']:>6.1f}%"
        )
    print(rule)


def _has_serialized(ctx: EdgeContext) -> bool:
    """True when all serialized engine directories exist under ``engine_root``."""
    return all((ctx.engine_root / name).exists() for name in _SERIALIZED_ENGINE_DIRS)


def _parity_modes(ctx: EdgeContext) -> set[str]:
    match ctx.parity_mode:
        case ParityMode.E2E:
            return {ParityMode.E2E.value}
        case ParityMode.ISOLATED:
            return {ParityMode.ISOLATED.value}
        case ParityMode.BOTH:
            return {ParityMode.E2E.value, ParityMode.ISOLATED.value}
    return {ParityMode.E2E.value}


def _run_inference(ctx: EdgeContext, mode: ExecutionMode, *, parity_active: str) -> None:
    ctx.execution_mode = mode
    ctx.parity_active = parity_active
    ctx.inference.seed = SEED
    ctx.stage_results.clear()
    InferencePipeline(get_inference_pipeline(ctx.profile.name)).run(ctx)
    ctx.parity_active = None


def _collect_stage_tensors(
    ctx: EdgeContext,
    pipeline_cfg,
    backend: str,
    parity_active: str,
) -> None:
    """Snapshot each stage's output tensors for later parity."""
    for stage_cfg in pipeline_cfg.stages:
        result = ctx.stage_results.get(stage_cfg.stage_id)
        if not result:
            continue
        tensors = result.get("tensors", {})
        if tensors:
            ctx.benchmark.record_stage_mode(
                backend, parity_active, stage_cfg.stage_name, tensors
            )


def _snapshot_reference(ctx: EdgeContext) -> None:
    """Stash eager e2e upstream tensors for isolated backend runs."""
    eager = ctx.benchmark.stage_tensors_by_mode.get("eager", {}).get(ParityMode.E2E.value, {})
    reference: dict[str, dict] = {}

    image_embs = eager.get("vision", {}).get("image_embs")
    if image_embs is not None:
        reference["vision"] = {"image_embs": image_embs}
    lm_hidden = eager.get("language", {}).get("lm_hidden")
    if lm_hidden is not None:
        reference["language"] = {"lm_hidden": lm_hidden}
    context_embs = eager.get("action_context", {}).get("context_embs")
    if context_embs is not None:
        reference["action_context"] = {"context_embs": context_embs}

    action_ref: dict = {}
    if ctx.inference.noise is not None:
        action_ref["initial_actions"] = ctx.inference.noise
    state = ctx.inference.action_side.get("state")
    if state is not None:
        action_ref["state"] = state
    embodiment_id = ctx.inference.action_side.get("embodiment_id")
    if embodiment_id is not None:
        action_ref["embodiment_id"] = embodiment_id
    if action_ref:
        reference["action"] = action_ref

    ctx.parity_reference = reference


def report_stage_parity(ctx: EdgeContext, reference: str = ExecutionMode.EAGER.value) -> None:
    by_mode = ctx.benchmark.stage_tensors_by_mode if ctx.benchmark else {}
    ref_e2e = by_mode.get(reference, {}).get(ParityMode.E2E.value, {})
    if not ref_e2e:
        return

    for parity_active in (ParityMode.E2E.value, ParityMode.ISOLATED.value):
        has_mode = any(
            parity_active in mode_map
            for backend, mode_map in by_mode.items()
            if backend != reference
        )
        if not has_mode:
            continue

        for backend, mode_map in by_mode.items():
            if backend == reference:
                continue
            stage_map = mode_map.get(parity_active)
            if not stage_map:
                continue

            rows: list[tuple[str, str, dict[str, float]]] = []
            for stage_name, tensor_name in STAGE_PARITY_TENSORS.items():
                a = ref_e2e.get(stage_name, {}).get(tensor_name)
                b = stage_map.get(stage_name, {}).get(tensor_name)
                if a is None or b is None:
                    continue
                rows.append((stage_name, tensor_name, parity_metrics(a, b)))

            if rows:
                _print_parity_table(f"{reference} ({parity_active})", backend, rows)


class BenchmarkPipeline:
    def __init__(self, config=None):
        self.config = config

    def run(self, ctx: EdgeContext) -> BenchmarkResult:
        if ctx.benchmark is None:
            ctx.benchmark = BenchmarkResult()

        pipeline_cfg = get_inference_pipeline(ctx.profile.name)
        parity_modes = _parity_modes(ctx)

        # Eager e2e establishes the reference path and upstream tensors.
        _run_inference(ctx, ExecutionMode.EAGER, parity_active=ParityMode.E2E.value)
        _collect_stage_tensors(ctx, pipeline_cfg, "eager", ParityMode.E2E.value)
        if ParityMode.ISOLATED.value in parity_modes:
            _snapshot_reference(ctx)

        for mode in (ExecutionMode.IN_MEMORY, ExecutionMode.SERIALIZED):
            if mode is ExecutionMode.SERIALIZED and not _has_serialized(ctx):
                continue

            backend = mode.value
            if ParityMode.E2E.value in parity_modes:
                _run_inference(ctx, mode, parity_active=ParityMode.E2E.value)
                _collect_stage_tensors(ctx, pipeline_cfg, backend, ParityMode.E2E.value)

            if ParityMode.ISOLATED.value in parity_modes:
                _run_inference(ctx, mode, parity_active=ParityMode.ISOLATED.value)
                _collect_stage_tensors(ctx, pipeline_cfg, backend, ParityMode.ISOLATED.value)

        report_stage_parity(ctx)
        return ctx.benchmark
