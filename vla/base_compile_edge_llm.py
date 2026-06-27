"""Shared Edge-LLM compile + benchmark runner for all VLAs.

``BaseEdgeCompileRunner`` implements the template-method lifecycle shared by every
VLA profile (groot, pi05, smolvla, molmoact2):

    load dataset frame → load policy → prepare compile inputs
    → export TRT (in-memory plugin modules and/or serialized .engine files)
    → optional C++ smoke test (profile hook)
    → benchmark PyTorch vs in-memory TRT vs serialized engines

Profile-specific behavior lives in ``VLAProfile`` subclasses and their export /
inference hooks; this file only orchestrates the common flow.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import torch.nn as nn

from trt.data import load_test_data
from trt.export.mode import ExportMode
from trt.export.pipeline import VLAExportPipeline
from trt.measure import mean, print_action_metrics, print_timing
from trt.serialize import load_serialized_modules
from trt.utils import load_policy
from vla.profile import InMemoryHandles, SerializedHandles, VLAProfile

# Default LeRobot dataset used to fetch a single observation frame for compile/benchmark.
DATASET_ID = "lerobot/libero"
# Fixed RNG seed for reproducible action sampling during export and inference.
SEED = 42
# ``Test/`` repo root — used to locate bundled C++ smoke-test binaries.
REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
# Default path to the generic Edge-LLM ``llm_inference`` binary (PI0.5 / SmolVLA / GR00T smoke).
DEFAULT_LLM_INFERENCE_BIN = (
    WORKSPACE_ROOT
    / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference"
)


class BaseEdgeCompileRunner:
    """Template-method runner: load → export → benchmark.

    Constructed by ``compile_edge_llm.main()`` after the profile is resolved from
    ``--model``. Holds the loaded policy, inner ``nn.Module``, and compile inputs
    for the duration of ``run()``.
    """

    def __init__(self, profile: VLAProfile, args: argparse.Namespace | None = None) -> None:
        self.profile = profile
        self.args = args or self.parse_args()
        # Fall back to CPU when CUDA is unavailable (export may still fail later).
        self.device = torch.device(
            self.args.device if torch.cuda.is_available() else "cpu"
        )
        # Populated during ``run()`` — kept as instance attrs for profile hooks.
        self.policy: Any = None
        self.model: nn.Module | None = None
        self.compile_inputs: dict[str, Any] | None = None
        self.tokenizer: Any = None

    @classmethod
    def build_arg_parser(cls, profile: VLAProfile) -> argparse.ArgumentParser:
        """Build the profile-specific CLI parser (phase 2 after ``--model``)."""
        parser = argparse.ArgumentParser(
            description=f"Export {profile.name} TensorRT engines for TensorRT-Edge-LLM",
        )

        # --- Model / data ---
        parser.add_argument(
            "--model-id",
            type=str,
            default=profile.model_id,
            help="HuggingFace model id or local checkpoint path.",
        )
        parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
        parser.add_argument(
            "--episode-index",
            type=int,
            default=0,
            help="LeRobot episode index for the benchmark frame.",
        )
        parser.add_argument(
            "--frame-index",
            type=int,
            default=0,
            help="Frame index within the episode.",
        )

        # --- Output paths ---
        parser.add_argument(
            "--engine-dir",
            type=str,
            default=profile.engine_dir_default,
            help="Root directory for serialized .engine files and sidecars.",
        )
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--llm-inference-bin",
            type=str,
            default=str(DEFAULT_LLM_INFERENCE_BIN),
            help="Path to TensorRT-Edge-LLM llm_inference binary for C++ smoke tests.",
        )

        # --- Export / inference tuning ---
        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=None,
            help="Override language max sequence length for TRT export.",
        )
        parser.add_argument(
            "--num-traj-samples",
            type=int,
            default=1,
            help="Number of action trajectory samples per inference call.",
        )
        parser.add_argument(
            "--max-generation-length",
            type=int,
            default=256,
            help="Max tokens for language generation (profiles that use it).",
        )

        # --- Mode switches ---
        parser.add_argument(
            "--export-only",
            action="store_true",
            help="Export serialized .engine files; skip in-memory TRT plugin compile.",
        )

        parser.add_argument(
            "--run-cpp-smoke",
            action="store_true",
            help="Run profile C++ smoke test after export (via ``post_export``).",
        )

        # --- Benchmark loop ---
        parser.add_argument("--num-iterations", type=int, default=12)
        parser.add_argument(
            "--warmup",
            type=int,
            default=3,
            help="Iterations excluded from summary averages.",
        )

        # Profile-specific flags (e.g. ``--groot-runtime-bin``).
        profile.add_arguments(parser)
        return parser

    @classmethod
    def parse_args_for_profile(
        cls,
        profile: VLAProfile,
        argv: list[str] | None = None,
    ) -> argparse.Namespace:
        """Parse CLI args for a known profile (called from ``compile_edge_llm``)."""
        return cls.build_arg_parser(profile).parse_args(argv)

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        """Parse CLI args using this runner's already-bound profile."""
        return self.parse_args_for_profile(self.profile, argv)

    def run(self) -> int:
        """Execute the full load → export → benchmark pipeline. Returns process exit code."""
        # Fetch one observation frame from LeRobot for compile trace inputs.
        data = load_test_data(
            dataset_id=self.args.dataset_id,
            episode_index=self.args.episode_index,
            frame_index=self.args.frame_index,
        )
        self.policy, self.model = self._load_policy()
        self.compile_inputs = self.profile.prepare_compile_inputs(
            policy=self.policy,
            data=data,
            device=self.device,
            args=self.args,
        )
        self.tokenizer = self.profile.get_tokenizer(policy=self.policy, args=self.args)
        
        print(
            f"model={self.profile.name}  dataset={self.args.dataset_id}  "
            f"episode={self.args.episode_index}  frame={self.args.frame_index}  "
            f"num_traj_samples={self.args.num_traj_samples}  "
            f"iters={self.args.num_iterations}  warmup={self.args.warmup}"
        )

        # Handles populated by export; used later in the benchmark loop.
        in_memory = InMemoryHandles()
        serialized_engine_root: str | None = None

        self._run_export(ExportMode.SERIALIZED)
        serialized_engine_root = self.args.engine_dir

        # Profile may run C++ smoke test and return early (non-zero exit on failure).
        exit_code = self.profile.post_export(self, serialized_engine_root)
        if exit_code is not None:
            return exit_code

        # Standard path: shared ``VLAExportPipeline`` (vision → language → action…).
        if not self.args.export_only:
            # Compile TRT modules held in RAM (AttentionPlugin path).
            in_memory = self._run_export(ExportMode.IN_MEMORY)
        else:
            return 0
            
        serialized = self._load_serialized_handles(serialized_engine_root)
        self._benchmark(in_memory, serialized)
        return 0

    def _load_policy(self) -> tuple[Any, nn.Module]:
        """Load the LeRobot policy wrapper and inner ``nn.Module``.

        Profiles may override ``load_policy`` when the default attribute paths
        (``policy._model`` vs ``policy.model``) differ.
        """
        if hasattr(self.profile, "load_policy"):
            return self.profile.load_policy(self.args.model_id, self.device)

        policy = load_policy(self.profile.policy_cls, self.args.model_id, self.device)
        policy = policy.to(self.device).eval()
        model = policy._model.to(self.device).eval()
        return policy, model

    def _run_export(self, mode: ExportMode) -> Any:
        """Run ``VLAExportPipeline`` in IN_MEMORY or SERIALIZED mode.

        Returns ``InMemoryHandles`` for in-memory export, or the raw
        ``PipelineResult`` for serialized export.
        """
        assert self.model is not None and self.compile_inputs is not None
        hooks = self.profile.make_export_hooks(
            tokenizer=self.tokenizer,
            args=self.args,
        )
        pipeline_cls = self.profile.export_pipeline_cls()
        result = pipeline_cls(hooks, io=self.profile.io).run(
            self.model,
            self.policy,
            self.device,
            self.compile_inputs,
            mode=mode,
            engine_root=self.args.engine_dir if mode is ExportMode.SERIALIZED else None,
            seed=SEED,
            max_seq_len=self.args.max_seq_len,
            accuracy_check=True,
        )
        if mode is ExportMode.IN_MEMORY:
            return InMemoryHandles(
                vision=result.handles.get("vision"),
                language=result.handles.get("language"),
                action=result.handles.get("action"),
                action_context=result.handles.get("action_context"),
            )
        return result

    def _load_serialized_handles(
        self,
        engine_root: str | None,
    ) -> SerializedHandles | None:
        """Load serialized TRT engines from ``engine_root`` via profile stage specs."""
        if engine_root is None:
            return None
        if not self.profile.serialized_stages:
            return SerializedHandles()

        specs = tuple(stage.to_module_spec() for stage in self.profile.serialized_stages)
        loaded = load_serialized_modules(engine_root, specs=specs)
        handles = SerializedHandles()
        for index, stage in enumerate(self.profile.serialized_stages):
            setattr(handles, stage.key, loaded[index])
        return handles

    def _benchmark(
        self,
        in_memory: InMemoryHandles,
        serialized: SerializedHandles | None,
    ) -> None:
        """Time PyTorch, in-memory TRT, and serialized engine inference over N iterations."""
        assert self.model is not None and self.compile_inputs is not None

        pt_times: list[float] = []
        trt_times: list[float] = []
        engine_times: list[float] = []
        action_ades: list[float] = []
        actionmean_abs: list[float] = []
        engine_action_ades: list[float] = []
        engine_actionmean_abs: list[float] = []

        # Optional fixed PyTorch reference for accuracy metrics (computed once before loop).
        pt_ref_for_trt = None
        trt_stage = getattr(self.profile, "in_memory_trt_stage", "vision")
        if (
            not getattr(self.profile, "prefer_same_iter_reference", False)
            and getattr(in_memory, trt_stage, None) is not None
        ):
            pt_ref_for_trt, _, _ = self.profile.run_inference_eager(
                self.model,
                self.policy,
                self.compile_inputs,
                seed=SEED,
                device=self.device,
                vision_module=in_memory.vision,
            )

        pt_ref_for_engine = None
        engine_stage = getattr(self.profile, "serialized_benchmark_stage", "vision")
        if (
            not getattr(self.profile, "prefer_same_iter_reference", False)
            and serialized is not None
            and getattr(serialized, engine_stage, None) is not None
        ):
            pt_ref_for_engine, _, _ = self.profile.run_inference_eager(
                self.model,
                self.policy,
                self.compile_inputs,
                seed=SEED,
                device=self.device,
                vision_module=serialized.vision,
            )

        # When True, compare TRT/engine output to PyTorch from the same iteration
        # (needed when stochastic action sampling differs run-to-run).
        prefer_same_iter = getattr(self.profile, "prefer_same_iter_reference", False)

        for iteration in range(self.args.num_iterations):
            print(f"\n=== iter {iteration} ===", flush=True)

            pred_actions_pt = None

            # --- PyTorch eager baseline ---
            if prefer_same_iter:
                elapsed, pred_actions_pt = self._timed_call_with_result(
                    lambda: self.profile.run_inference_eager(
                        self.model,
                        self.policy,
                        self.compile_inputs,
                        seed=SEED,
                        device=self.device,
                    )
                )
            else:
                elapsed = self._timed_call(
                    lambda: self.profile.run_inference_eager(
                        self.model,
                        self.policy,
                        self.compile_inputs,
                        seed=SEED,
                        device=self.device,
                    )
                )
            pt_times.append(elapsed)
            print(f"  PyTorch    : {elapsed:7.1f} ms")

            # --- In-memory TRT plugin modules ---
            if (
                not self.args.export_only
                and getattr(in_memory, trt_stage, None) is not None
            ):
                elapsed, pred = self._timed_call_with_result(
                    lambda: self.profile.run_inference_trt(
                        self.model,
                        self.policy,
                        self.compile_inputs,
                        handles=in_memory,
                        seed=SEED,
                        device=self.device,
                    )
                )
                trt_times.append(elapsed)
                trt_ref = pred_actions_pt if prefer_same_iter else pt_ref_for_trt
                if trt_ref is not None:
                    metrics = self.profile.compute_action_metrics(
                        pred,
                        trt_ref,
                        self.policy,
                    )
                    action_ades.append(metrics["action_ade"])
                    actionmean_abs.append(metrics["mean_abs"])
                    print(
                        f"  TRT Plugin : {elapsed:7.1f} ms   "
                        f"actionADE={metrics['action_ade']:.6f}  "
                        f"mean_abs={metrics['mean_abs']:.6f}"
                    )
                else:
                    print(f"  TRT Plugin : {elapsed:7.1f} ms")

            # --- Serialized .engine files loaded from disk ---
            if not self.args.export_only and serialized is not None:
                elapsed, pred = self._timed_call_with_result(
                    lambda: self.profile.run_inference_trt(
                        self.model,
                        self.policy,
                        self.compile_inputs,
                        handles=serialized,
                        seed=SEED,
                        device=self.device,
                    )
                )
                engine_times.append(elapsed)
                engine_ref = pred_actions_pt if prefer_same_iter else pt_ref_for_engine
                if engine_ref is not None:
                    metrics = self.profile.compute_action_metrics(
                        pred,
                        engine_ref,
                        self.policy,
                    )
                    engine_action_ades.append(metrics["action_ade"])
                    engine_actionmean_abs.append(metrics["mean_abs"])
                    print(
                        f"  Serialized : {elapsed:7.1f} ms   "
                        f"actionADE={metrics['action_ade']:.6f}  "
                        f"mean_abs={metrics['mean_abs']:.6f}"
                    )
                else:
                    print(f"  Serialized : {elapsed:7.1f} ms")

        self._print_summary(
            pt_times,
            trt_times,
            engine_times,
            action_ades,
            actionmean_abs,
            engine_action_ades,
            engine_actionmean_abs,
        )

    def _sync_and_time(self, fn) -> float:
        """Run ``fn`` and return elapsed wall time in ms (CUDA-synchronized)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - start)

    def _timed_call(self, fn) -> float:
        """Time a void inference call; discard the return value."""
        return self._sync_and_time(fn)

    def _timed_call_with_result(self, fn) -> tuple[float, torch.Tensor]:
        """Time an inference call and return ``(elapsed_ms, pred_actions)``."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        # Profile inference functions return ``(actions, side_dict, extra_ms)``.
        return elapsed, result[0]

    def _print_summary(
        self,
        pt_times: list[float],
        trt_times: list[float],
        engine_times: list[float],
        action_ades: list[float],
        actionmean_abs: list[float],
        engine_action_ades: list[float],
        engine_actionmean_abs: list[float],
    ) -> None:
        """Print averaged timing and accuracy metrics after warmup iterations."""
        warmup = self.args.warmup
        print("\n" + "=" * 78)
        print(f"Summary  (warmup={warmup} / {self.args.num_iterations})")
        print("=" * 78)

        if pt_times:
            print_timing(self.profile.display_name, pt_times[warmup:])
        if trt_times:
            print_timing("TRT Plugin FP16", trt_times[warmup:])
        if engine_times:
            print_timing("Serialized Engine", engine_times[warmup:])
        if action_ades:
            print_action_metrics("TRT Action ADE", action_ades[warmup:])
            print_action_metrics("TRT Action mean abs", actionmean_abs[warmup:])
        if engine_action_ades:
            print_action_metrics("Engine Action ADE", engine_action_ades[warmup:])
            print_action_metrics("Engine Action mean abs", engine_actionmean_abs[warmup:])

        if pt_times and trt_times:
            pt_avg = mean(pt_times[warmup:])
            trt_avg = mean(trt_times[warmup:])
            speedup = pt_avg / trt_avg if trt_avg > 0 else float("nan")
            print(
                f"\n  Speedup (TRT vs PyTorch): {speedup:5.2f}x   "
                f"({pt_avg:.1f} -> {trt_avg:.1f} ms)"
            )
        if pt_times and engine_times:
            pt_avg = mean(pt_times[warmup:])
            engine_avg = mean(engine_times[warmup:])
            speedup = pt_avg / engine_avg if engine_avg > 0 else float("nan")
            print(
                f"  Speedup (Engine vs PyTorch): {speedup:5.2f}x   "
                f"({pt_avg:.1f} -> {engine_avg:.1f} ms)"
            )
