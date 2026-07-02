from __future__ import annotations

from trt.config.execution_mode import ExecutionMode
from trt.context import EdgeContext

SEED = 42


def _run_inference(ctx: EdgeContext, mode: ExecutionMode) -> None:
    from trt.config.pipeline_registry import get_inference_pipeline
    from trt.pipelines.inference import InferencePipeline

    ctx.execution_mode = mode
    ctx.inference.seed = SEED
    ctx.stage_results.clear()
    model_type = getattr(ctx.profile, "pipeline_model_type", None) or ctx.profile.name
    InferencePipeline(get_inference_pipeline(model_type)).run(ctx)


def _has_in_memory(ctx: EdgeContext) -> bool:
    return ctx.handles.in_memory.vision is not None


def _has_serialized(ctx: EdgeContext) -> bool:
    return ctx.handles.serialized.vision is not None


def report_language_logits_parity(ctx: EdgeContext) -> None:
    if ctx.handles.serialized.language is None or ctx.inference.image_embs is None:
        return

    from trt.executor.models.groot.inference.language import compare_language_logits

    print("\nLanguage logits parity (serialized TRT vs eager):")
    try:
        metrics = compare_language_logits(ctx, print_metrics=True)
        print(
            f"  summary               mean_abs={metrics['mean_abs']:.6f}  "
            f"max_abs={metrics['max_abs']:.6f}  rel_l2={metrics['relative_l2']:.6f}"
        )
    except Exception as exc:
        print(f"  skipped: {exc}")


def report_groot_benchmark(ctx: EdgeContext) -> None:
    report_language_logits_parity(ctx)
    report_action_parity(ctx)


def report_action_parity(ctx: EdgeContext) -> None:
    result = ctx.benchmark
    if result is None:
        return

    reference = result.actions.get("pytorch")
    if reference is None:
        return

    from trt.measure import compute_action_parity_metrics

    print("Action parity vs pytorch:")
    for name, pred in result.actions.items():
        if name == "pytorch":
            continue
        metrics = compute_action_parity_metrics(pred, reference)
        print(
            f"  {name:<22} ade={metrics['action_ade']:.6f}  "
            f"fde={metrics['action_fde']:.6f}  "
            f"mean_abs={metrics['mean_abs']:.6f}  "
            f"max_abs={metrics['max_abs']:.6f}"
        )
