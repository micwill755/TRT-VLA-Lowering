"""Fine-grained Torch-TRT compile stage timing for VLA e2e comparisons.

Buckets wall time into the stages we care about for one-shot vs stock:

1. Export / AOT capture  (timed by the caller around export)
2. ``run_decompositions`` (second AOT walk in stock; skipped in one-shot)
3. Post-lowering + partitioning
4. TensorRT engine build

Patches both defining modules and ``torch_tensorrt.dynamo._compiler``'s
bound imports so hooks actually fire on the compile path.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class StageSnapshot:
    """Accumulated seconds / call counts since install or last reset."""

    timings_seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    hooked: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def get(self, label: str) -> float:
        return float(self.timings_seconds.get(label, 0.0))

    def count(self, label: str) -> int:
        return int(self.counts.get(label, 0))

    def buckets(self, *, export_seconds: float = 0.0) -> dict[str, float]:
        """Non-overlapping-ish stage buckets for printing."""
        decomp = self.get("export.ExportedProgram.run_decompositions")
        pre = self.get("compiler.pre_export_lowering")
        post = self.get("compiler.post_lowering")
        constant_fold = self.get("post_lowering_pass.constant_fold") or self.get(
            "lowering.constant_fold"
        )
        complex_graph = self.get("post_lowering_pass.complex_graph_detection")
        partition = (
            self.get("partitioning.fast_partition")
            or self.get("partitioning.global_partition")
            or self.get("partitioning.hierarchical_adjacency_partition")
        )
        engine_build = self.get("trt.Builder.build_engine_with_config") or self.get(
            "trt.Builder.build_serialized_network"
        )
        convert = self.get("compiler.convert_module") or self.get(
            "conversion.convert_module"
        )
        # convert includes network construct + engine build; isolate build.
        convert_minus_build = max(0.0, convert - engine_build) if convert else 0.0
        return {
            "export_aot": float(export_seconds),
            "run_decompositions": decomp,
            "pre_export_lowering": pre,
            "post_lowering": post,
            "constant_fold": constant_fold,
            "complex_graph_detection": complex_graph,
            "partition": partition,
            "post_lowering_partition": post + partition,
            "convert_minus_engine_build": convert_minus_build,
            "engine_build": engine_build,
        }


class CompileStageTimer:
    def __init__(self) -> None:
        self._timings: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._originals: list[tuple[Any, str, Any]] = []
        self._hooked: list[str] = []
        self._missing: list[str] = []
        self._installed = False

    def reset(self) -> None:
        self._timings.clear()
        self._counts.clear()

    def snapshot(self) -> StageSnapshot:
        return StageSnapshot(
            timings_seconds=dict(self._timings),
            counts=dict(self._counts),
            hooked=list(self._hooked),
            missing=list(self._missing),
        )

    def _wrap(self, owner: Any, attr: str, label: str) -> bool:
        if not hasattr(owner, attr):
            self._missing.append(label)
            return False
        original = getattr(owner, attr)
        if not callable(original):
            self._missing.append(f"{label} (not callable)")
            return False

        timer = self

        def wrapped(*args: Any, **kwargs: Any):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timer._timings[label] += time.perf_counter() - t0
                timer._counts[label] += 1

        setattr(owner, attr, wrapped)
        self._originals.append((owner, attr, original))
        self._hooked.append(label)
        return True

    def _wrap_pass_manager(self, manager: Any, prefix: str) -> bool:
        """Mutate DynamoPassManager.passes in place (same pattern as Flux)."""
        passes = getattr(manager, "passes", None)
        if passes is None:
            self._missing.append(f"{prefix}.* (no .passes)")
            return False
        if not isinstance(passes, list):
            self._missing.append(
                f"{prefix}.* (.passes not a list: {type(passes).__name__})"
            )
            return False

        timer = self
        original_passes = list(passes)
        wrapped: list[Callable[..., Any]] = []
        for index, pass_fn in enumerate(passes):
            name = getattr(pass_fn, "__name__", f"pass_{index}")
            label = f"{prefix}.{name}"

            def make_wrapper(fn: Callable[..., Any], lbl: str) -> Callable[..., Any]:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.perf_counter()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        timer._timings[lbl] += time.perf_counter() - t0
                        timer._counts[lbl] += 1

                wrapper.__name__ = getattr(fn, "__name__", lbl)
                wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
                if hasattr(fn, "_lowering_pass_config"):
                    wrapper._lowering_pass_config = fn._lowering_pass_config  # type: ignore[attr-defined]
                return wrapper

            wrapped.append(make_wrapper(pass_fn, label))
            self._hooked.append(label)

        manager.passes[:] = wrapped
        self._originals.append((manager, "passes", original_passes))
        return True

    def install(self) -> StageSnapshot:
        if self._installed:
            return self.snapshot()

        try:
            from torch_tensorrt.dynamo.lowering.passes import (
                _aten_lowering_pass as aten_lowering,
            )
            from torch_tensorrt.dynamo.lowering.passes import constant_folding
        except Exception as exc:
            self._missing.append(f"lowering import: {exc}")
        else:
            self._wrap_pass_manager(
                aten_lowering.ATEN_PRE_LOWERING_PASSES, "pre_lowering_pass"
            )
            self._wrap_pass_manager(
                aten_lowering.ATEN_POST_LOWERING_PASSES, "post_lowering_pass"
            )
            self._wrap(aten_lowering, "pre_export_lowering", "lowering.pre_export_lowering")
            self._wrap(aten_lowering, "post_lowering", "lowering.post_lowering")
            self._wrap(constant_folding, "constant_fold", "lowering.constant_fold")
            folder = getattr(constant_folding, "_TorchTensorRTConstantFolder", None)
            if folder is not None and hasattr(folder, "run"):
                self._wrap(folder, "run", "lowering.constant_folder.run")

        try:
            from torch_tensorrt.dynamo import _compiler
        except Exception as exc:
            self._missing.append(f"_compiler import: {exc}")
        else:
            for name, label in (
                ("pre_export_lowering", "compiler.pre_export_lowering"),
                ("post_lowering", "compiler.post_lowering"),
                ("convert_module", "compiler.convert_module"),
                ("compile_module", "compiler.compile_module"),
            ):
                self._wrap(_compiler, name, label)

        try:
            from torch.export import ExportedProgram
        except Exception as exc:
            self._missing.append(f"ExportedProgram import: {exc}")
        else:
            self._wrap(
                ExportedProgram,
                "run_decompositions",
                "export.ExportedProgram.run_decompositions",
            )

        try:
            from torch_tensorrt.dynamo import partitioning
        except Exception as exc:
            self._missing.append(f"partitioning import: {exc}")
        else:
            for name, label in (
                ("fast_partition", "partitioning.fast_partition"),
                ("global_partition", "partitioning.global_partition"),
                (
                    "hierarchical_adjacency_partition",
                    "partitioning.hierarchical_adjacency_partition",
                ),
            ):
                self._wrap(partitioning, name, label)

        try:
            from torch_tensorrt.dynamo.conversion import _conversion
        except Exception as exc:
            self._missing.append(f"conversion import: {exc}")
        else:
            self._wrap(_conversion, "convert_module", "conversion.convert_module")

        try:
            import tensorrt as trt
        except Exception as exc:
            self._missing.append(f"tensorrt import: {exc}")
        else:
            builder = getattr(trt, "Builder", None)
            if builder is None:
                self._missing.append("trt.Builder")
            else:
                for attr, label in (
                    (
                        "build_engine_with_config",
                        "trt.Builder.build_engine_with_config",
                    ),
                    (
                        "build_serialized_network",
                        "trt.Builder.build_serialized_network",
                    ),
                ):
                    self._wrap(builder, attr, label)

        self._installed = True
        return self.snapshot()

    def uninstall(self) -> None:
        for owner, attr, original in reversed(self._originals):
            setattr(owner, attr, original)
        self._originals.clear()
        self._installed = False


@contextmanager
def stage_timing() -> Iterator[CompileStageTimer]:
    timer = CompileStageTimer()
    timer.install()
    try:
        yield timer
    finally:
        timer.uninstall()


def print_stage_breakdown(
    label: str,
    *,
    export_seconds: float,
    compile_seconds: float,
    snapshot: StageSnapshot,
) -> dict[str, float]:
    buckets = snapshot.buckets(export_seconds=export_seconds)
    # Non-overlapping wall buckets only (nested pass timings are reported separately).
    accounted_keys = (
        "export_aot",
        "run_decompositions",
        "pre_export_lowering",
        "post_lowering",
        "partition",
        "convert_minus_engine_build",
        "engine_build",
    )
    accounted = sum(buckets[k] for k in accounted_keys)
    other = max(0.0, export_seconds + compile_seconds - accounted)
    buckets["other_unaccounted"] = other

    decomp_n = snapshot.count("export.ExportedProgram.run_decompositions")
    folder_run = snapshot.get("lowering.constant_folder.run")
    print(f"\n=== [{label}] stage breakdown ===")
    print(
        f"  {'export / AOT capture':32s} {buckets['export_aot']:7.3f}s"
    )
    print(
        f"  {'run_decompositions':32s} {buckets['run_decompositions']:7.3f}s"
        f"  (count={decomp_n})"
    )
    print(
        f"  {'pre_export_lowering':32s} {buckets['pre_export_lowering']:7.3f}s"
    )
    print(
        f"  {'post_lowering (total)':32s} {buckets['post_lowering']:7.3f}s"
    )
    print(
        f"    {'constant_fold':30s} {buckets['constant_fold']:7.3f}s"
        f"  (folder.run={folder_run:.3f}s)"
    )
    print(
        f"    {'complex_graph_detection':30s} {buckets['complex_graph_detection']:7.3f}s"
    )
    print(
        f"  {'partition':32s} {buckets['partition']:7.3f}s"
    )
    print(
        f"  {'convert (ex-engine-build)':32s} {buckets['convert_minus_engine_build']:7.3f}s"
    )
    print(
        f"  {'TensorRT engine build':32s} {buckets['engine_build']:7.3f}s"
    )
    print(
        f"  {'other / unaccounted':32s} {buckets['other_unaccounted']:7.3f}s"
    )
    print(
        f"  {'export+compile wall':32s} {export_seconds + compile_seconds:7.3f}s"
    )
    # Per-pass table for the rest of post_lowering (helps Naren's "fuse cheap passes" idea).
    pass_rows = sorted(
        (
            (k, v)
            for k, v in snapshot.timings_seconds.items()
            if k.startswith("post_lowering_pass.")
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if pass_rows:
        print("  post_lowering passes (ranked):")
        for name, seconds in pass_rows:
            short = name.removeprefix("post_lowering_pass.")
            print(f"    {short:30s} {seconds:7.3f}s  (n={snapshot.count(name)})")
    if snapshot.missing:
        print(f"  (missing hooks: {', '.join(snapshot.missing[:6])})")
    return buckets


__all__ = [
    "CompileStageTimer",
    "StageSnapshot",
    "print_stage_breakdown",
    "stage_timing",
]
