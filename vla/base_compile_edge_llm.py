"""Shared skeleton for VLA Edge-LLM export scripts.

Subclasses set class attributes (model id, default engine dir, etc.) and implement
model-specific hooks. The common LeRobot path is:

  load_test_data -> load_policy -> prepare_policy_batch -> export/benchmark
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import argparse
import pathlib
import torch

from trt.data import (
    load_test_data, 
    prepare_policy_batch
)
from trt.vision import (
    save_visual_engine_for_edge_llm
    DEFAULT_VISION_TRT_SETTINGS, 
    VisionEngineSpec
)

from trt.language import (
    save_language_engine_for_edge_llm 
)
from trt.tokenizer import (
    save_embedding_table,
    save_tokenizer_for_edge_llm
)

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_ID = "lerobot/libero"
SEED = 42
DEFAULT_LLM_INFERENCE_BIN = (
    WORKSPACE_ROOT / "gitlab/TensorRT-Edge-LLM/build-plugin-trt11/examples/llm/llm_inference"
)

class BaseEdgeBuilder(ABC):
    """Template-method base for Edge-LLM export scripts."""

    name: ClassVar[str] = "vla"
    model_id: ClassVar[str] = ""
    engine_dir_default: ClassVar[str] = "/tmp/vla_edge_llm"
    fill_missing_cameras: ClassVar[bool] = False

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        self.args = args or self.parse_args()
        self.device = torch.device(
            self.args.device if torch.cuda.is_available() else "cpu"
        )
        self.data: dict[str, Any] | None = None
        self.policy: Any = None
        self.model: Any = None

    @classmethod
    def parse_args(cls, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"Export {cls.name} TensorRT engines for TensorRT-Edge-LLM",
        )
        parser.add_argument("--model-id", type=str, default=cls.model_id)
        parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
        parser.add_argument("--episode-index", type=int, default=0)
        parser.add_argument("--frame-index", type=int, default=0)
        parser.add_argument("--engine-dir", type=str, default=cls.engine_dir_default)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--llm-inference-bin",
            type=str,
            default=str(DEFAULT_LLM_INFERENCE_BIN),
            help="Path to TensorRT-Edge-LLM llm_inference binary for C++ smoke tests.",
        )
        parser.add_argument("--seed", type=int, default=SEED)
        parser.add_argument("--max-seq-len", type=int, default=None)
        parser.add_argument("--num-traj-samples", type=int, default=1)
        parser.add_argument("--max-generation-length", type=int, default=256)
        parser.add_argument(
            "--export-only",
            action="store_true",
            help="Export serialized .engine files; skip in-memory TRT plugin compile.",
        )
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--no-accuracy-check", action="store_true")
        parser.add_argument("--no-stage-parity", action="store_true")
        parser.add_argument("--run-cpp-smoke", action="store_true")
        parser.add_argument("--skip-export", action="store_true")
        parser.add_argument("--skip-pytorch", action="store_true")
        parser.add_argument("--skip-trt", action="store_true")
        parser.add_argument("--skip-engine", action="store_true")
        parser.add_argument("--num-iterations", type=int, default=12)
        parser.add_argument("--warmup", type=int, default=3)
        cls.add_arguments(parser)
        return parser.parse_args(argv)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Override to register model-specific CLI flags."""

    def load_data(self) -> dict[str, Any]:
        self.data = load_test_data(
            dataset_id=self.args.dataset_id,
            episode_index=self.args.episode_index,
            frame_index=self.args.frame_index,
        )
        return self.data

    @abstractmethod
    def load_policy(self) -> Any:
        """Load ``self.policy`` and ``self.model``."""

    self.compile_trt_with_plugin(self):
        # TODO build a seperate obj for managin in memory engine compile for eager comparison

        # init engines to None
        '''trt_vision = trt_lm = trt_diffusion = plugin_info = None
        serialized_engine_info = None
        edge_plugin_info = None

        if not args.skip_trt and not args.export_only:
            trt_vision, trt_lm, trt_diffusion, plugin_info = self.compile_trt_with_plugin(
                model,
                policy,
                device,
                compile_inputs,
                seed=args.seed,
                max_generation_length=args.max_generation_length,
                num_traj_samples=args.num_traj_samples,
                max_seq_len=args.max_seq_len,
                debug=args.debug,
                accuracy_check=not args.no_accuracy_check,
            )'''

        print(
            "SmolVLA in-memory TRT plugin compile is not wired in this script; "
            "use Test/compile/smolvla_compile_trt.py or pass --skip-trt."
        )

    @abstractmethod
    def pre_process_action_input(self) -> dict:
        """action input for some vlas require additional, e.g. groot state and embodiment id."""

    @abstractmethod
    def build_smolvla_vision_export_params(
        core: nn.Module,
        pixel_values: torch.Tensor,
        device: torch.device,
        *,
        io: PipelineIOSpec,
        trt_settings: dict | None,
    ) -> VisionEngineSpec:
        """each vla will have to load in these based on hf config"""

    @abstractmethod
    def create_image_embs(image_embed_shape) -> torch.Tensor:
        "image embds is different to VLA eg either [B * S, H] or [B, S, H]"

    @abstractmethod
    def create_packed_embs(image_embed_shape) -> torch.Tensor:
        "everyone must define the prefill input for LM as inputs_embeds [B,L,H]"

    def save_edge_engines_for_edge_llm(
        model: nn.Module,
        policy: Any,
        device: str,
        model_inputs: dict,
        *,
        seed: int = 42,
        offload_module_to_cpu: bool = False,
        max_generation_length: int = 256,
        num_traj_samples: int = 1,
        max_seq_len: int | None = None,
        hidden: int | None = None,
        debug: bool = False,
        accuracy_check: bool = True,
        engine_root: str,
        io: PipelineIOSpec,
        tokenizer,
    ) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, dict]:
        # create local engine path
        engine_root = str(pathlib.Path(engine_root))

        # every model input has the following
        tokenized_data = model_inputs['tokenized_data']
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']

        # --------- pre process for action input ---------
        self.pre_process_action_input()

        # -------------------------
        # Vision engine
        # -------------------------
        print("compiling vision")
        engine_dir = str(pathlib.Path(engine_root) / "visual")

        vis_params = self.build_vision_export_params(
            model,
            pixel_values,
            device,
            io=io,
            trt_settings=VISION_TRT_SETTINGS,
            input_dtype=torch.float16,
        )

        save_visual_engine_for_edge_llm(
            pixel_values,
            engine_dir,
            vis_params,
            device=device,
        )
        # VitRunner visual.engine output (flat): image_embed_flat_shape == [B*S, H]
        image_embs = self.create_image_embs()

        # --------- pack text + image embs together to create 1 tensor of multimodal embeds ---------

        # llm_inference requires tokenizer + embedding table + chat template in language/.
        print("saving tokenizer")
        language_engine_dir = str(pathlib.Path(engine_root) / "language")
        language_model = model.backbone.eagle_model.language_model
        save_embedding_table(language_model, language_engine_dir)
        save_tokenizer_for_edge_llm(
            language_engine_dir,
            tokenizer=tokenizer,
            chat_template=build_groot_vitrunner_chat_template(tokenizer),
        )

        # -------------------------
        # Language engine
        # -------------------------
        print("compiling language")

        spec = build_groot_language_export_params(
            model,
            input_ids,
            image_token_id=int(vis_params.image_token_id),
            seq_len_per_image=int(vis_params.config_seq_len),
            device=torch.device(device),
            io=io,
            dtype=torch.float16,
        )
        save_language_engine_for_edge_llm(language_engine_dir, spec)

        mtmdl_embds = pack_groot_language_inputs(
            model,
            trt_image_embs,
            input_ids,
            attention_mask,
        )





    def run(self) -> int:
        self.load_data()
        self.load_policy()
        self.prepare_inputs()
        # TODO: do we still need an in memory compile option?
        # self.compile_trt_with_plugin():

        compile_inputs = prepare_policy_batch(
            self.policy,
            self.data,
            self.device,
            self.args.model_id,
            fill_missing=self.fill_missing_cameras,
        )
        
        return self.export_and_benchmark()

    @abstractmethod
    def export_and_benchmark(self) -> int:
        """Compile/export engines and run parity/timing loops."""
