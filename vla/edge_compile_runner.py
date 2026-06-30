from __future__ import annotations

import argparse
import time

import torch

from trt.config.pipeline_registry import get_pipeline_for_profile
from trt.data import load_test_data
from trt.measure import mean, print_action_metrics, print_timing
from trt.pipeline import VLAExportPipeline
from trt.serialize import load_serialized_modules
from trt.utils import load_plugins_for_trt, load_policy
from vla.profile import InMemoryHandles, SerializedHandles, VLAProfile

SEED = 42

class EdgeCompileRunner:
    def __init__(self, device, profile: VLAProfile, args: argparse.Namespace):
        self.profile = profile
        self.args = args
        self.device = device
        self.policy = None
        self.model = None
        self.model_inputs = None

    @classmethod
    def build_arg_parser(cls, profile: VLAProfile) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        p.add_argument("--model-id", default=None)
        p.add_argument("--dataset-id", default=None)
        p.add_argument("--episode-index", type=int, default=0)
        p.add_argument("--frame-index", type=int, default=0)
        p.add_argument("--engine-dir", default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--max-seq-len", type=int, default=None)
        p.add_argument("--num-traj-samples", type=int, default=1)
        p.add_argument("--export-only", action="store_true")
        p.add_argument("--run-cpp-smoke", action="store_true")
        p.add_argument("--num-iterations", type=int, default=12)
        p.add_argument("--warmup", type=int, default=3)
        profile.add_arguments(p)
        return p

    def run(self) -> int:
        load_plugins_for_trt()
        data = load_test_data(
            self.args.dataset_id,
            self.args.episode_index,
            self.args.frame_index,
        )
        self.policy = self._load_policy()
        self.model = self.policy._model.to(self.device).eval()
        self.model_inputs = self.profile.prepare_compile_inputs(
            policy=self.policy,
            data=data,
            device=self.device,
            args=self.args,
        )

        config = get_pipeline_for_profile(self.profile)
        ctx = VLAExportPipeline(config).run(
            profile=self.profile,
            policy=self.policy,
            device=self.device,
            model_inputs=self.model_inputs,
            engine_root=self.args.engine_dir,
        )

        if code := self.profile.post_export(self, self.args.engine_dir):
            return code
        if self.args.export_only:
            return 0

        serialized = self._load_serialized(self.args.engine_dir)
        self._benchmark(ctx, serialized)
        return 0

    def _load_policy(self):
        if hasattr(self.profile, "load_policy"):
            return self.profile.load_policy(self.args.model_id, self.device)
        policy = load_policy(self.profile.policy_cls, self.args.model_id, self.device)
        policy = policy.to(self.device).eval()
        return policy

    def _load_serialized(self, engine_root: str | None) -> SerializedHandles | None:
        if not engine_root or not self.profile.serialized_stages:
            return SerializedHandles()
        specs = tuple(s.to_module_spec() for s in self.profile.serialized_stages)
        loaded = load_serialized_modules(engine_root, specs=specs)
        handles = SerializedHandles()
        for i, stage in enumerate(self.profile.serialized_stages):
            setattr(handles, stage.key, loaded[i])
        return handles

    def _benchmark(self, ctx, serialized: SerializedHandles | None) -> None:
        in_memory = InMemoryHandles(
            vision=ctx.handles.get("vision"),
            language=ctx.handles.get("language"),
            action=ctx.handles.get("action"),
            action_context=ctx.handles.get("action_context"),
        )
        pt_times, trt_times, eng_times = [], [], []
        for i in range(self.args.num_iterations):
            t0 = time.perf_counter()
            pt_actions, _, _ = self.profile.run_inference_eager(
                self.model, self.policy, self.model_inputs,
                seed=SEED, device=self.device,
            )
            pt_times.append(time.perf_counter() - t0)

            if in_memory.vision:
                t0 = time.perf_counter()
                self.profile.run_inference_trt(
                    self.model, self.policy, self.model_inputs,
                    handles=in_memory, seed=SEED, device=self.device,
                )
                trt_times.append(time.perf_counter() - t0)

            if serialized and serialized.vision:
                t0 = time.perf_counter()
                self.profile.run_inference_trt(
                    self.model, self.policy, self.model_inputs,
                    handles=serialized, seed=SEED, device=self.device,
                )
                eng_times.append(time.perf_counter() - t0)

        w = self.args.warmup
        print_timing("pytorch", mean(pt_times[w:]))
        if trt_times:
            print_timing("in-memory trt", mean(trt_times[w:]))
        if eng_times:
            print_timing("serialized", mean(eng_times[w:]))