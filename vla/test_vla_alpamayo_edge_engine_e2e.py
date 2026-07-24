"""Eager vs Edge ONNX→TRT engine parity for Alpamayo (action_inference ABI).

Mirrors ``test_vla_alpamayo_e2e.py`` stage structure (vision → language → action),
but loads prebuilt Edge engines instead of compiling with Torch-TensorRT:

  engines/visual/visual.engine
  engines/llm/llm.engine
  engines/action/action.engine

Each stage feeds the next from engine outputs (same pattern as the Torch-TRT
e2e), and compares against HF eager on the same stage inputs.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_position_ids,
)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

_ALPAMAYO_SRC = _TEST_ROOT.parent / "alpamayo" / "src"
if _ALPAMAYO_SRC.is_dir() and str(_ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(_ALPAMAYO_SRC))

from alpamayo_r1 import helper

from trt.measure import parity
from trt.modules.export.alpamayo_language import build_alpamayo_prefix_embs
from trt.modules.export.alpamayo_lm_plugin import (
    build_alpamayo_rope_cache,
    pack_deepstack_to_ds_stack,
)
from trt.modules.export.alpamayo_vision import VisualFixedGrid
from trt.modules.export.diffusion import (
    AlpamayoPrefixKVStepEncoderExportModule,
    StaticActionVelocityStepExportModule,
)
from trt.serialize import _trt_dtype_to_torch
from trt.utils import force_hf_attention, free_cuda_memory

DEFAULT_ENGINE_DIR = (
    Path.home() / "tensorrt-edgellm-workspace/Alpamayo-R1-10B/engines"
)
DEFAULT_MODEL_PATH = str(Path.home() / "tensorrt-edgellm-workspace/Alpamayo-R1-10B")
DEFAULT_PLUGIN = (
    "/home/micwilliams/workspace/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
)


class EdgeEngine:
    """Minimal TensorRT runner for Edge ``*.engine`` files (no Test config schema)."""

    def __init__(self, engine_path: Path, *, profile: int = 0):
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create context for {self.engine_path}")

        self.profile = int(profile)
        if self.engine.num_optimization_profiles > 0:
            ok = self.context.set_optimization_profile_async(
                self.profile, torch.cuda.current_stream().cuda_stream
            )
            if ok is False:
                raise RuntimeError(
                    f"Failed to select optimization profile {self.profile} "
                    f"for {self.engine_path}"
                )

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        self._zero_dummies: dict[torch.dtype, torch.Tensor] = {}

    def close(self) -> None:
        """Release TensorRT runtime buffers so the next engine can deserialize."""
        self.context = None
        self.engine = None
        self.runtime = None
        self._zero_dummies.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _zero_binding(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        dummy = self._zero_dummies.get(dtype)
        if dummy is None or dummy.device != device:
            dummy = torch.zeros(1, device=device, dtype=dtype)
            self._zero_dummies[dtype] = dummy
        return dummy

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise KeyError(f"{self.engine_path.name} missing inputs: {missing}")

        first = next(iter(inputs.values()))
        if first.device.type != "cuda":
            raise RuntimeError("EdgeEngine requires CUDA tensors")
        device = first.device

        bound: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            tensor = inputs[name].to(device=device).contiguous()
            ok = self.context.set_input_shape(name, tuple(tensor.shape))
            if ok is False:
                raise RuntimeError(
                    f"set_input_shape failed for {name}: {tuple(tensor.shape)}"
                )
            bound[name] = (
                self._zero_binding(tensor.dtype, device)
                if tensor.numel() == 0
                else tensor
            )

        # Edge action/LLM engines update KV caches in-place: present_* must alias
        # the corresponding input cache buffer (same data_ptr).
        alias_map = {
            "present_k_cache_": "k_cache_",
            "present_v_cache_": "v_cache_",
            "present_key_values_": "past_key_values_",
        }

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            aliased = None
            for present_prefix, cache_prefix in alias_map.items():
                if name.startswith(present_prefix):
                    suffix = name[len(present_prefix) :]
                    cache_name = f"{cache_prefix}{suffix}"
                    if cache_name in bound:
                        aliased = bound[cache_name]
                    break
            if aliased is not None:
                outputs[name] = aliased
                continue
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved output shape for {name}: {shape}")
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device=device, dtype=dtype)

        for name, tensor in bound.items():
            if self.context.set_tensor_address(name, tensor.data_ptr()) is False:
                raise RuntimeError(f"Failed to bind input {name}")
        for name, tensor in outputs.items():
            if self.context.set_tensor_address(name, tensor.data_ptr()) is False:
                raise RuntimeError(f"Failed to bind output {name}")

        stream = torch.cuda.current_stream(device).cuda_stream
        if self.context.execute_async_v3(stream_handle=stream) is False:
            raise RuntimeError(f"execute_async_v3 failed for {self.engine_path}")
        return outputs


def _load_edge_plugin(plugin_path: str) -> None:
    import ctypes

    path = Path(plugin_path)
    if path.exists():
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _cuda_time_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def prepare_edge_vision_inputs(
    visual,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build QwenViTRunner / visual.engine bindings from HF visual helpers."""
    grid_thw = grid_thw.to(device=device)
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    rotary = visual.rotary_pos_emb(position_ids)
    rotary = rotary.reshape(pixel_values.shape[0], -1).to(dtype=torch.float32)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device=device)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    max_seqlen = max(int(x) for x in lengths) if lengths else 1
    max_seqlen_carrier = torch.zeros(max_seqlen, device=device, dtype=torch.int32)

    idx, weight = get_vision_bilinear_indices_and_weights(
        grid_thw,
        num_grid_per_side=int(visual.num_grid_per_side),
        spatial_merge_size=int(visual.spatial_merge_size),
    )
    return {
        "input": pixel_values.contiguous(),
        "rotary_pos_emb": rotary.contiguous(),
        "cu_seqlens": cu_seqlens.contiguous(),
        "max_seqlen_carrier": max_seqlen_carrier.contiguous(),
        "fast_pos_embed_idx": idx.to(device=device, dtype=torch.int64).contiguous(),
        "fast_pos_embed_weight": weight.to(device=device, dtype=dtype).contiguous(),
    }


def load_config(device, model_path: str, dtype=torch.float16):
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Run with the Alpamayo/lerobot env and PYTHONPATH including alpamayo/src."
        ) from exc

    model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to(device).eval()
    processor = helper.get_processor(model.tokenizer)
    return model, processor


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eager vs Edge ONNX→TRT engine accuracy for Alpamayo"
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--engine-dir",
        default=str(DEFAULT_ENGINE_DIR),
        help="Edge engine tree with visual/, llm/, action/ subdirs",
    )
    parser.add_argument("--plugin", default=DEFAULT_PLUGIN)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--kv-capacity",
        type=int,
        default=4096,
        help="LM/action KV cache capacity matching Edge builder",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.float16
    engine_root = Path(args.engine_dir).resolve()
    kv_capacity = int(args.kv_capacity)

    os.environ.setdefault("EDGE_LLM_PLUGIN_SO", args.plugin)
    _load_edge_plugin(args.plugin)

    model, processor = load_config(device, args.model_path, dtype=dtype)
    vlm_model = model.vlm.model
    vision = vlm_model.visual.to(device=device, dtype=dtype).eval()
    language = vlm_model.language_model.to(device=device, dtype=dtype).eval()
    model.expert = model.expert.to(device=device, dtype=dtype).eval()

    force_hf_attention(vision, "eager", use_cache=False)
    force_hf_attention(language, "eager")
    force_hf_attention(model.expert, "eager")

    try:
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "physical_ai_av is required; use the Alpamayo Python environment."
        ) from exc

    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    tokenized_data = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": tokenized_data,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        str(device),
    )
    pixel_values = model_inputs["tokenized_data"]["pixel_values"].to(
        device=device, dtype=dtype
    )
    image_grid_thw = model_inputs["tokenized_data"]["image_grid_thw"].to(device)

    # ---------------------------------------------------------------------------
    # STEP 1 — vision
    # ---------------------------------------------------------------------------
    print("Loading Edge visual.engine")
    visual_eager = VisualFixedGrid(vision, image_grid_thw).to(device=device).eval()
    with torch.no_grad():
        embs_eager, deepstack_eager = visual_eager(pixel_values, None)

    vision_inputs = prepare_edge_vision_inputs(
        vision, pixel_values, image_grid_thw, device, dtype
    )
    vision_engine = EdgeEngine(engine_root / "visual" / "visual.engine")
    vision_out = vision_engine(vision_inputs)
    embs_edge = vision_out["output"]
    deepstack_edge = [
        vision_out[f"deepstack_features_{i}"]
        for i in range(3)
        if f"deepstack_features_{i}" in vision_out
    ]

    vision_eager_ms = _cuda_time_ms(
        lambda: visual_eager(pixel_values, None),
        warmup=args.warmup,
        iters=args.iters,
    )
    vision_edge_ms = _cuda_time_ms(
        lambda: vision_engine(vision_inputs),
        warmup=args.warmup,
        iters=args.iters,
    )

    parity("vision A vs Edge", embs_eager, embs_edge)
    for idx, (a, b) in enumerate(zip(deepstack_eager, deepstack_edge)):
        parity(f"vision deepstack[{idx}]", a, b)

    # Keep Edge vision outputs; drop the visual engine + HF visual before LM.
    embs_edge = embs_edge.detach().contiguous()
    deepstack_edge = [t.detach().contiguous() for t in deepstack_edge]
    vision_engine.close()
    free_cuda_memory(
        vision_engine,
        visual_eager,
        embs_eager,
        deepstack_eager,
        pixel_values,
        vision_inputs,
        vision_out,
    )
    vision.cpu()
    vlm_model.visual.cpu()
    model.expert.cpu()
    free_cuda_memory()

    # ---------------------------------------------------------------------------
    # STEP 2 — language prefill
    # ---------------------------------------------------------------------------
    print("Running eager language, then loading Edge llm.engine")

    tokenized = copy.deepcopy(model_inputs["tokenized_data"])
    input_ids = tokenized.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )
    image_token_id = int(vlm_model.config.image_token_id)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        inputs_embeds, visual_pos_masks = build_alpamayo_prefix_embs(
            language.embed_tokens,
            input_ids,
            embs_edge,
            image_token_id=image_token_id,
        )
        attn = tokenized.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        try:
            position_ids, rope_deltas = vlm_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw=None, attention_mask=attn
            )
        except (TypeError, IndexError):
            mm_token_type_ids = (input_ids == image_token_id).int()
            position_ids, rope_deltas = vlm_model.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attn,
            )

    bsz = int(inputs_embeds.shape[0])
    seq_len = int(inputs_embeds.shape[1])
    cfg = language.config
    hidden_size = int(cfg.hidden_size)
    num_key_value_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_layers = len(getattr(language, "model", language).layers)
    lm_head = model.vlm.lm_head.to(device=device, dtype=dtype).eval()

    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype).contiguous()
    ds_for_hf = []
    for de in deepstack_edge:
        de = de.to(device=device, dtype=dtype)
        if de.dim() > 2:
            de = de.reshape(-1, de.shape[-1])
        ds_for_hf.append(de)

    def _run_eager_language():
        return language(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=ds_for_hf,
            use_cache=False,
            return_dict=True,
        )

    eager_out = _run_eager_language()
    lm_hidden_eager = eager_out.last_hidden_state
    eager_logits = lm_head(lm_hidden_eager[:, -1:]).float().contiguous()
    lm_eager_ms = _cuda_time_ms(
        _run_eager_language, warmup=args.warmup, iters=args.iters
    )
    free_cuda_memory(eager_out, lm_hidden_eager)

    ds_stack = pack_deepstack_to_ds_stack(
        deepstack_edge,
        visual_pos_masks,
        batch_size=bsz,
        max_seq_len=seq_len,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
    rope_rotary_cos_sin = build_alpamayo_rope_cache(
        language,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        seq_len=seq_len,
        max_seq_len=kv_capacity,
        head_dim=head_dim,
        device=device,
    )

    # Bake action mRoPE before moving the HF LM off GPU (~16GB).
    n_diffusion_tokens = int(model.action_space.get_action_space_dims()[0])
    action_space_dims = tuple(int(x) for x in model.action_space.get_action_space_dims())
    prefix_len = seq_len
    rope_delta = int(rope_deltas.reshape(-1)[0].item())
    action_base = rope_delta + prefix_len
    action_abs = (
        torch.arange(n_diffusion_tokens, device=device) + action_base
    ).view(1, 1, -1).expand(3, bsz, -1).long()
    rotary_emb = getattr(language, "rotary_emb", None)
    if rotary_emb is None:
        rotary_emb = getattr(getattr(language, "model", None), "rotary_emb", None)
    if rotary_emb is None:
        raise RuntimeError("language model has no rotary_emb for action mRoPE")
    cos, sin = rotary_emb(torch.ones(1, device=device, dtype=dtype), action_abs)
    h2 = head_dim // 2
    action_rope = torch.cat(
        [cos[:, :n_diffusion_tokens, :h2].float(), sin[:, :n_diffusion_tokens, :h2].float()],
        dim=-1,
    ).contiguous()
    attention_pos_id = (
        torch.arange(n_diffusion_tokens, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(bsz, -1)
        .contiguous()
    )
    step_position_ids = action_abs.clone()

    # HF 8B LM and Edge llm.engine cannot both reside in 32GB; offload HF first.
    language.cpu()
    lm_head.cpu()
    model.vlm.cpu()
    free_cuda_memory(ds_for_hf, deepstack_edge, embs_edge, language, lm_head)
    free_cuda_memory()

    context_lengths = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    # Edge fresh-prefill ABI: empty kvcache_start_index selects profile-0 empty binding.
    kvcache_start_index = torch.empty(0, dtype=torch.int32, device=device)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.int64)
    past_key_values = {
        f"past_key_values_{i}": torch.zeros(
            bsz,
            2,
            num_key_value_heads,
            kv_capacity,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for i in range(num_layers)
    }
    lm_inputs = {
        "inputs_embeds": inputs_embeds,
        **past_key_values,
        "rope_rotary_cos_sin": rope_rotary_cos_sin,
        "context_lengths": context_lengths,
        "kvcache_start_index": kvcache_start_index,
        "last_token_ids": last_token_ids,
        **{
            f"deepstack_embeds_{i}": ds_stack[i]
            for i in range(ds_stack.shape[0])
        },
    }

    print("Loading Edge llm.engine (prefill profile)")
    lm_engine = EdgeEngine(engine_root / "llm" / "llm.engine", profile=0)
    lm_out = lm_engine(lm_inputs)
    edge_logits = lm_out["logits"].float()
    lm_edge_ms = _cuda_time_ms(
        lambda: lm_engine(lm_inputs), warmup=args.warmup, iters=args.iters
    )
    parity("language logits A vs Edge", eager_logits, edge_logits)

    present_kv = {
        i: lm_out[f"present_key_values_{i}"].detach().cpu()
        for i in range(num_layers)
    }
    lm_engine.close()
    free_cuda_memory(
        lm_engine,
        lm_out,
        lm_inputs,
        past_key_values,
        inputs_embeds,
        ds_stack,
        rope_rotary_cos_sin,
        eager_logits,
        edge_logits,
    )
    free_cuda_memory()

    # ---------------------------------------------------------------------------
    # STEP 3 — action / diffusion step
    # ---------------------------------------------------------------------------
    print("Running eager action, then loading Edge action.engine")

    model.expert.to(device=device, dtype=dtype)
    model.action_in_proj.to(device=device, dtype=dtype)
    model.action_out_proj.to(device=device, dtype=dtype)
    force_hf_attention(model.expert, "eager")

    # PREFIX_KV handoff: copy LM present caches into full-capacity action buffers.
    k_caches = []
    v_caches = []
    prefix_k_stack = []
    prefix_v_stack = []
    for i in range(num_layers):
        present = present_kv[i].to(device=device, dtype=dtype)
        k_full = torch.zeros(
            bsz, num_key_value_heads, kv_capacity, head_dim, device=device, dtype=dtype
        )
        v_full = torch.zeros_like(k_full)
        k_full[:, :, :prefix_len] = present[:, 0, :, :prefix_len]
        v_full[:, :, :prefix_len] = present[:, 1, :, :prefix_len]
        k_caches.append(k_full)
        v_caches.append(v_full)
        prefix_k_stack.append(present[:, 0, :, :prefix_len].contiguous())
        prefix_v_stack.append(present[:, 1, :, :prefix_len].contiguous())
    free_cuda_memory(present_kv)

    prefix_k = torch.stack(prefix_k_stack, dim=0)
    prefix_v = torch.stack(prefix_v_stack, dim=0)
    free_cuda_memory(prefix_k_stack, prefix_v_stack)

    diffusion_model = StaticActionVelocityStepExportModule(
        step_encoder=AlpamayoPrefixKVStepEncoderExportModule(model),
        action_expert=model.expert,
        velocity_decoder=model.action_out_proj,
        output_tokens=n_diffusion_tokens,
        cast_hidden_fp32=False,
    ).eval().to(device=device, dtype=dtype)

    step_actions = torch.randn(bsz, *action_space_dims, device=device, dtype=dtype)
    step_timestep = torch.zeros(bsz, 1, 1, device=device, dtype=dtype)
    step_attention_mask = torch.zeros(
        bsz,
        1,
        n_diffusion_tokens,
        prefix_len + n_diffusion_tokens,
        device=device,
        dtype=dtype,
    )
    diffusion_input = (
        step_actions,
        step_timestep,
        prefix_k,
        prefix_v,
        step_position_ids,
        step_attention_mask,
    )
    eager_velocity = diffusion_model(*diffusion_input)
    # Edge action engine returns Euler-integrated denoised trajectory.
    dt = torch.full((bsz, 1, 1), 0.1, device=device, dtype=torch.float32)
    t0 = torch.zeros(bsz, device=device, dtype=torch.float32)
    t1 = torch.full((bsz,), 0.1, device=device, dtype=torch.float32)
    eager_denoised = (
        step_actions.float() + dt * eager_velocity.float()
    ).contiguous()
    action_eager_ms = _cuda_time_ms(
        lambda: diffusion_model(*diffusion_input),
        warmup=args.warmup,
        iters=args.iters,
    )

    # Offload HF expert before deserializing the action engine.
    free_cuda_memory(diffusion_model, eager_velocity, prefix_k, prefix_v)
    model.expert.cpu()
    model.action_in_proj.cpu()
    model.action_out_proj.cpu()
    free_cuda_memory()

    action_inputs = {
        "noise_trajectory": step_actions.float().contiguous(),
        "time_steps_t0": t0.contiguous(),
        "time_steps_t1": t1.contiguous(),
        "kvcache_start_index": torch.full(
            (bsz,), prefix_len, device=device, dtype=torch.int32
        ),
        "rope_rotary_cos_sin": action_rope,
        "attention_pos_id": attention_pos_id,
        **{f"k_cache_{i}": k_caches[i] for i in range(num_layers)},
        **{f"v_cache_{i}": v_caches[i] for i in range(num_layers)},
    }
    print("Loading Edge action.engine")
    action_engine = EdgeEngine(engine_root / "action" / "action.engine")
    action_out = action_engine(action_inputs)
    edge_denoised = action_out["denoised_trajectory"].float()
    action_edge_ms = _cuda_time_ms(
        lambda: action_engine(action_inputs),
        warmup=args.warmup,
        iters=args.iters,
    )
    parity("action denoised A vs Edge", eager_denoised, edge_denoised)

    def _speedup(eager_ms: float, edge_ms: float) -> str:
        return f"{(eager_ms / edge_ms):.3f}x" if edge_ms > 0 else "n/a"

    print(f"vision eager execute: {vision_eager_ms:.3f} ms")
    print(f"vision edge execute: {vision_edge_ms:.3f} ms")
    print(f"vision speedup: {_speedup(vision_eager_ms, vision_edge_ms)}")
    print(f"lm eager execute: {lm_eager_ms:.3f} ms")
    print(f"lm edge execute: {lm_edge_ms:.3f} ms")
    print(f"lm speedup: {_speedup(lm_eager_ms, lm_edge_ms)}")
    print(f"action eager execute: {action_eager_ms:.3f} ms")
    print(f"action edge execute: {action_edge_ms:.3f} ms")
    print(f"action speedup: {_speedup(action_eager_ms, action_edge_ms)}")
    eager_total = vision_eager_ms + lm_eager_ms + action_eager_ms
    edge_total = vision_edge_ms + lm_edge_ms + action_edge_ms
    print(f"total eager execute: {eager_total:.3f} ms")
    print(f"total edge execute: {edge_total:.3f} ms")
    print(f"total speedup: {_speedup(eager_total, edge_total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
