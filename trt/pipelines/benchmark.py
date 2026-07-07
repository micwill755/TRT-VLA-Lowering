from __future__ import annotations

from trt.context import BenchmarkResult, EdgeContext
from trt.config.execution_mode import ExecutionMode
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

def _run_inference(ctx: EdgeContext, mode: ExecutionMode) -> None:
    ctx.execution_mode = mode
    ctx.inference.seed = SEED
    ctx.stage_results.clear()
    InferencePipeline(get_inference_pipeline(ctx.profile.name)).run(ctx)

def _collect_stage_tensors(ctx: EdgeContext, pipeline_cfg, backend: str) -> None:
    """Snapshot each stage's output tensors under ``backend`` for later parity."""
    for stage_cfg in pipeline_cfg.stages:
        result = ctx.stage_results.get(stage_cfg.stage_id)
        if not result:
            continue
        tensors = result.get("tensors", {})
        if tensors:
            ctx.benchmark.record_stage(backend, stage_cfg.stage_name, tensors)

def report_stage_parity(ctx: EdgeContext, reference: str = ExecutionMode.EAGER.value) -> None:
    stage_tensors = ctx.benchmark.stage_tensors if ctx.benchmark else {}
    ref = stage_tensors.get(reference)
    if not ref:
        return

    for backend, stage_map in stage_tensors.items():
        if backend == reference:
            continue
        rows: list[tuple[str, str, dict[str, float]]] = []
        for stage_name, tensor_name in STAGE_PARITY_TENSORS.items():
            a = ref.get(stage_name, {}).get(tensor_name)
            b = stage_map.get(stage_name, {}).get(tensor_name)
            if a is None or b is None:
                continue
            rows.append((stage_name, tensor_name, parity_metrics(a, b)))
        if rows:
            _print_parity_table(reference, backend, rows)

class BenchmarkPipeline:
    def __init__(self, config=None):
        self.config = config

    def run(self, ctx: EdgeContext) -> BenchmarkResult:
        if ctx.benchmark is None:
            ctx.benchmark = BenchmarkResult()

        pipeline_cfg = get_inference_pipeline(ctx.profile.name)

        for mode in (ExecutionMode.EAGER, ExecutionMode.IN_MEMORY, ExecutionMode.SERIALIZED):
            if mode is ExecutionMode.SERIALIZED and not _has_serialized(ctx):
                continue
            _run_inference(ctx, mode)
            _collect_stage_tensors(ctx, pipeline_cfg, mode.value)

        report_stage_parity(ctx)
        return ctx.benchmark
