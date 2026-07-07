from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type

import torch

from trt.config.pipeline_registry import get_benchmark_pipeline, get_export_pipeline
from trt.context import EdgeContext
from trt.data import load_test_data
from trt.pipelines.benchmark import BenchmarkPipeline
from trt.pipelines.export import ExportPipeline
from trt.pipelines.load import LoadPipeline
from trt.profile import VLAProfile
from trt.plugin.plugin_utils import load_plugins_for_trt

DATASET_ID = "lerobot/libero"

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
}

class EdgeOrchestrator:
    def __init__(self, profile: Type[VLAProfile], args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.profile = profile(self.device, args.model_id)

    @staticmethod
    def build_arg_parser(profile: Type[VLAProfile]) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=f"Export and benchmark {profile.display_name} TensorRT Edge-LLM engines",
        )
        parser.add_argument("--model-id", default=profile.model_id)
        parser.add_argument("--dataset-id", default=DATASET_ID)
        parser.add_argument("--episode-index", type=int, default=0)
        parser.add_argument("--frame-index", type=int, default=0)
        parser.add_argument("--engine-dir", default=profile.engine_dir_default)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--max-seq-len", type=int, default=None)
        parser.add_argument("--num-iterations", type=int, default=12)
        parser.add_argument("--warmup", type=int, default=3)
        parser.add_argument(
            "--export-only",
            action="store_true",
            help="Compile serialized engines to --engine-dir; skip benchmark.",
        )
        parser.add_argument(
            "--benchmark-only",
            action="store_true",
            help="Load engines from --engine-dir and benchmark; skip export.",
        )
        return parser

    def run(self) -> int:
        if self.args.export_only and self.args.benchmark_only:
            raise SystemExit("Use only one of --export-only or --benchmark-only")

        load_plugins_for_trt()
        ctx = self._build_context()

        if self._should_export():
            export_cfg = get_export_pipeline(self.profile.name)
            ExportPipeline(export_cfg).run(ctx)

        if self._should_benchmark():
            LoadPipeline.for_model_type(self.profile.name).run(ctx)
            bench_cfg = get_benchmark_pipeline(self.profile.name)
            BenchmarkPipeline(bench_cfg).run(ctx)

        return 0

    def _should_export(self) -> bool:
        return not self.args.benchmark_only

    def _should_benchmark(self) -> bool:
        return not self.args.export_only

    def _build_context(self) -> EdgeContext:
        data = load_test_data(
            self.args.dataset_id,
            episode_index=self.args.episode_index,
            frame_index=self.args.frame_index,
        )
        model_inputs = self.profile.prepare_compile_inputs(
            data=data,
            args=self.args,
        )
        return EdgeContext(
            profile=self.profile,
            policy=self.profile.policy,
            model=self.profile.model,
            device=self.device,
            model_inputs=model_inputs,
            engine_root=Path(self.args.engine_dir),
            args=self.args,
            trt_settings=TRT_SETTINGS
        )
