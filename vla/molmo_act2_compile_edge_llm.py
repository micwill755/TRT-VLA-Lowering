"""Export MolmoAct2 TensorRT engines for TensorRT-Edge-LLM-style deployment."""

from __future__ import annotations

from typing import Any

from lerobot.policies.molmoact2 import MolmoAct2Policy

from trt.utils import load_policy
from vla.base_compile_edge_llm import BaseEdgeBuilder


class MolmoAct2EdgeBuilder(BaseEdgeBuilder):
    name = "MolmoAct2"
    model_id = "allenai/MolmoAct2"
    engine_dir_default = "/tmp/molmoact2_edge_llm"
    fill_missing_cameras = False

    def load_policy(self) -> Any:
        self.policy = load_policy(MolmoAct2Policy, self.args.model_id, self.device)
        self.model = self.policy.model
        return self.policy

    def export_and_benchmark(self) -> int:
        # TODO: vision / language / action TRT export and parity loops.
        print(
            f"model={self.args.model_id}  dataset={self.args.dataset_id}  "
            f"episode={self.args.episode_index}  frame={self.args.frame_index}  "
            f"compile_inputs keys={sorted(self.compile_inputs)}"
        )
        return 0


def main() -> int:
    return MolmoAct2EdgeBuilder().run()


if __name__ == "__main__":
    raise SystemExit(main())
