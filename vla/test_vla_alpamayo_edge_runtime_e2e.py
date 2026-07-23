from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
import tensorrt as trt
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.compile import patch_trt_interpreter_output_names
from trt.plugin.attention import ContextAttentionMaskType
from trt.plugin.plugin_utils import (
    load_plugins_for_trt,
    patch_language_attention,
    restore_attention,
)
from trt.utils import force_hf_attention, free_cuda_memory


torch_tensorrt.logging.set_level(logging.WARNING)

TRT_SETTINGS = {
    "disable_tf32": False,
    "use_fp32_acc": False,
    "use_explicit_typing": False,
    "truncate_double": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
    "enabled_precisions": {torch.float16},
}

LANGUAGE_TRT_SETTINGS = {
    "enabled_precisions": {torch.float32},
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "disable_tf32": True,
    "min_block_size": 1,
    "decompose_attention": True,
    "offload_module_to_cpu": True,
}

VISION_PROFILES = {
    "input": ((640, 1536), (37184, 1536), (73728, 1536)),
    "rotary_pos_emb": ((640, 36), (37184, 36), (73728, 36)),
    "cu_seqlens": ((2,), (116,), (116,)),
    "max_seqlen_carrier": ((1,), (384,), (768,)),
    "fast_pos_embed_idx": ((4, 640), (4, 37184), (4, 73728)),
    "fast_pos_embed_weight": ((4, 640), (4, 37184), (4, 73728)),
}


@torch.library.custom_op("trt::alpamayo_action_attention", mutates_args=())
def alpamayo_action_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position_ids: torch.Tensor,
    write_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del key, value, rope_cos, rope_sin, position_ids, write_indices, valid_lengths
    return query.clone(), k_cache.clone(), v_cache.clone()


@alpamayo_action_attention.register_fake
def _alpamayo_action_attention_fake(
    query,
    key,
    value,
    k_cache,
    v_cache,
    rope_cos,
    rope_sin,
    position_ids,
    write_indices,
    valid_lengths,
):
    del key, value, rope_cos, rope_sin, position_ids, write_indices, valid_lengths
    return torch.empty_like(query), torch.empty_like(k_cache), torch.empty_like(v_cache)


@dynamo_tensorrt_converter(
    torch.ops.trt.alpamayo_action_attention.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def _convert_alpamayo_action_attention(
    ctx: ConversionContext, target, args, kwargs, name
):
    del target, kwargs
    tensors = [
        value
        if isinstance(value, trt.ITensor)
        else get_trt_tensor(ctx, value, f"{name}_i{index}")
        for index, value in enumerate(args)
    ]
    q, k, v, k_cache, v_cache, cos, sin, position_ids, write_indices, valid = tensors

    q_rope = ctx.net.add_rotary_embedding(q, cos, sin, True, int(q.shape[-1]))
    k_rope = ctx.net.add_rotary_embedding(k, cos, sin, True, int(k.shape[-1]))
    if q_rope is None or k_rope is None:
        raise RuntimeError("Failed to create TensorRT RotaryEmbedding layers")
    q_rope.set_input(3, position_ids)
    k_rope.set_input(3, position_ids)

    present_k = ctx.net.add_kv_cache_update(
        k_cache, k_rope.get_output(0), write_indices, trt.KVCacheMode.LINEAR
    )
    present_v = ctx.net.add_kv_cache_update(
        v_cache, v, write_indices, trt.KVCacheMode.LINEAR
    )
    if present_k is None or present_v is None:
        raise RuntimeError("Failed to create TensorRT KVCacheUpdate layers")

    attention = ctx.net.add_attention(
        q_rope.get_output(0),
        present_k.get_output(0),
        present_v.get_output(0),
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    if attention is None:
        raise RuntimeError("Failed to create TensorRT Attention layer")
    attention.key_value_lengths = valid
    attention.decomposable = True
    return (
        attention.get_output(0),
        present_k.get_output(0),
        present_v.get_output(0),
    )


def _make_profiled_input(
    name: str,
    dtype: torch.dtype,
    profile_shapes: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> torch_tensorrt.Input:
    return torch_tensorrt.Input(
        name=name,
        dtype=dtype,
        profiles=[
            {
                "min_shape": minimum,
                "opt_shape": optimum,
                "max_shape": maximum,
            }
            for minimum, optimum, maximum in profile_shapes
        ],
    )


def _allow_empty_profile_min(spec: torch_tensorrt.Input, profile_index: int = 0) -> None:
    """Allow Edge-LLM's empty fresh-prefill kvcache_start_index binding.

    Torch-TensorRT's public Input validation rejects a zero profile dimension,
    while TensorRT's AttentionPlugin and the Edge ONNX builder support the
    [0]-element binding. Build the normal [1] spec first, then make the profile
    match LLMEngineRunner's profile-0 ABI before the TRT interpreter consumes it.
    """

    spec.profiles[profile_index]["min_shape"] = (0,)
    spec.shape["min_shape"] = (0,)


def _as_tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class EdgeRuntimeVision(nn.Module):
    """Qwen3-VL vision graph with action_inference/QwenViTRunner's exact ABI."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _attention(
        self,
        attn: nn.Module,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden.shape[0]
        q, k, v = (
            attn.qkv(hidden)
            .reshape(seq_len, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        original_dtype = q.dtype
        q = (q.float() * cos.unsqueeze(-2)) + (
            self._rotate_half(q.float()) * sin.unsqueeze(-2)
        )
        k = (k.float() * cos.unsqueeze(-2)) + (
            self._rotate_half(k.float()) * sin.unsqueeze(-2)
        )
        output = torch.ops.trt.vit_attention_plugin.default(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            cu_seqlens,
            max_seqlen_carrier,
            int(attn.num_heads),
            int(q.shape[-1]),
        )
        output = output.reshape(seq_len, -1).to(original_dtype)
        return attn.proj(output)

    def forward(
        self,
        input: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,
        fast_pos_embed_weight: torch.Tensor,
    ):
        hidden_states = input
        hidden_states = self.visual.patch_embed(hidden_states)
        pos = F.embedding(fast_pos_embed_idx, self.visual.pos_embed.weight)
        pos = pos * fast_pos_embed_weight[:, :, None].to(pos.dtype)
        hidden_states = hidden_states + pos.sum(dim=0).to(hidden_states.dtype)

        rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = rotary.cos(), rotary.sin()
        deepstack = []
        deepstack_indexes = list(self.visual.deepstack_visual_indexes)

        for layer_num, block in enumerate(self.visual.blocks):
            residual = hidden_states
            normed = block.norm1(hidden_states)
            hidden_states = residual + self._attention(
                block.attn,
                normed,
                cu_seqlens,
                max_seqlen_carrier,
                cos,
                sin,
            )
            residual = hidden_states
            hidden_states = residual + block.mlp(block.norm2(hidden_states))
            if layer_num in deepstack_indexes:
                idx = deepstack_indexes.index(layer_num)
                deepstack.append(self.visual.deepstack_merger_list[idx](hidden_states))

        output = self.visual.merger(hidden_states)
        return (output, *deepstack)


class _EdgeRuntimeLMBase(nn.Module):
    def __init__(self, decoder: nn.Module, lm_head: nn.Module, num_deepstack: int):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.num_deepstack = int(num_deepstack)

    def _forward_impl(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        deepstack_embeds: tuple[torch.Tensor, ...],
    ):
        hidden = inputs_embeds.to(next(self.decoder.parameters()).dtype)
        present = []
        for layer_idx, layer in enumerate(self.decoder.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, layer_present = layer.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                past_key_value=past_key_values[layer_idx],
                ctx_len=context_lengths,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = residual + _as_tensor(hidden)
            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = residual + _as_tensor(layer.mlp(hidden))
            if layer_idx < self.num_deepstack:
                hidden = hidden + deepstack_embeds[layer_idx].to(hidden.dtype)
            present.append(layer_present)

        hidden = _as_tensor(self.decoder.norm(hidden))
        # Edge runtime uses batch=1 for Alpamayo. index_select is supported by
        # the Torch-TensorRT Dynamo converter and preserves the [B, 1, H] shape.
        selected = torch.index_select(hidden, 1, last_token_ids.reshape(-1))
        logits = self.lm_head(selected).float()
        return (logits, *present)


def make_edge_runtime_lm(
    decoder: nn.Module,
    lm_head: nn.Module,
    num_layers: int,
    num_deepstack: int,
) -> nn.Module:
    past_names = [f"past_key_values_{i}" for i in range(num_layers)]
    ds_names = [f"deepstack_embeds_{i}" for i in range(num_deepstack)]
    parameters = (
        ["inputs_embeds"]
        + past_names
        + [
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
        ]
        + ds_names
    )
    past_tuple = "(" + ", ".join(past_names) + ",)"
    ds_tuple = "(" + ", ".join(ds_names) + ",)"
    source = (
        f"def forward(self, {', '.join(parameters)}):\n"
        f"    return self._forward_impl(inputs_embeds, {past_tuple}, "
        "rope_rotary_cos_sin, context_lengths, kvcache_start_index, "
        f"last_token_ids, {ds_tuple})\n"
    )
    namespace: dict = {}
    exec(source, namespace)
    module = _EdgeRuntimeLMBase(decoder, lm_head, num_deepstack)
    module.forward = namespace["forward"].__get__(module, type(module))
    return module


class _EdgeRuntimeActionBase(nn.Module):
    def __init__(self, model, decoder: nn.Module, num_layers: int):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.decoder = decoder
        self.num_layers = int(num_layers)
        self.num_waypoints = int(model.action_space.get_action_space_dims()[0])

    def _attention(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ):
        batch, seq_len, _ = hidden.shape
        num_heads = int(attention.config.num_attention_heads)
        num_kv_heads = int(attention.config.num_key_value_heads)
        head_dim = int(attention.head_dim)
        q = attention.q_proj(hidden).reshape(batch, seq_len, num_heads, head_dim)
        k = attention.k_proj(hidden).reshape(
            batch, seq_len, num_kv_heads, head_dim
        )
        v = attention.v_proj(hidden).reshape(
            batch, seq_len, num_kv_heads, head_dim
        )
        if getattr(attention, "q_norm", None) is not None:
            q = attention.q_norm(q)
        if getattr(attention, "k_norm", None) is not None:
            k = attention.k_norm(k)
        q = q.transpose(1, 2).to(torch.float16)
        k = k.transpose(1, 2).to(torch.float16)
        v = v.transpose(1, 2).to(torch.float16)
        q = q * (head_dim**-0.5)
        half = head_dim // 2
        cos = rope_rotary_cos_sin[0, :, :half].reshape(-1, half).to(torch.float16)
        sin = rope_rotary_cos_sin[0, :, half:].reshape(-1, half).to(torch.float16)
        valid_lengths = kvcache_start_index + self.num_waypoints
        output, present_k, present_v = (
            torch.ops.trt.alpamayo_action_attention.default(
                q,
                k,
                v,
                k_cache,
                v_cache,
                cos,
                sin,
                position_ids,
                kvcache_start_index,
                valid_lengths,
            )
        )
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return attention.o_proj(output.to(hidden.dtype)), present_k, present_v

    def _forward_impl(
        self,
        noise_trajectory: torch.Tensor,
        time_steps_t0: torch.Tensor,
        time_steps_t1: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
        k_caches: tuple[torch.Tensor, ...],
        v_caches: tuple[torch.Tensor, ...],
    ):
        batch = noise_trajectory.shape[0]
        noise_half = noise_trajectory.to(torch.float16)
        timestep = time_steps_t0.view(batch, 1, 1).to(torch.float16)
        hidden = self.action_in_proj(noise_half, timestep)
        if hidden.ndim == 2:
            hidden = hidden.reshape(batch, self.num_waypoints, -1)
        present_k = []
        present_v = []

        for layer_idx, layer in enumerate(self.decoder.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, layer_present_k, layer_present_v = self._attention(
                layer.self_attn,
                hidden,
                k_caches[layer_idx],
                v_caches[layer_idx],
                rope_rotary_cos_sin,
                attention_pos_id,
                kvcache_start_index,
            )
            hidden = residual + _as_tensor(hidden)
            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = residual + _as_tensor(layer.mlp(hidden))
            present_k.append(layer_present_k)
            present_v.append(layer_present_v)

        hidden = _as_tensor(self.decoder.norm(hidden))
        velocity = self.action_out_proj(hidden[:, -self.num_waypoints :])
        velocity = velocity.reshape(batch, self.num_waypoints, 2).float()
        dt = (time_steps_t1 - time_steps_t0).view(batch, 1, 1)
        denoised = noise_trajectory + dt * velocity
        return (denoised, *present_k, *present_v)


def make_edge_runtime_action(model, decoder: nn.Module, num_layers: int) -> nn.Module:
    k_names = [f"k_cache_{i}" for i in range(num_layers)]
    v_names = [f"v_cache_{i}" for i in range(num_layers)]
    leading = [
        "noise_trajectory",
        "time_steps_t0",
        "time_steps_t1",
        "kvcache_start_index",
        "rope_rotary_cos_sin",
        "attention_pos_id",
    ]
    parameters = leading + k_names + v_names
    source = (
        f"def forward(self, {', '.join(parameters)}):\n"
        "    return self._forward_impl("
        + ", ".join(leading)
        + ", ("
        + ", ".join(k_names)
        + ",), ("
        + ", ".join(v_names)
        + ",))\n"
    )
    namespace: dict = {}
    exec(source, namespace)
    module = _EdgeRuntimeActionBase(model, decoder, num_layers)
    module.forward = namespace["forward"].__get__(module, type(module))
    return module


def _export_engine(
    module: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    input_specs: list[torch_tensorrt.Input],
    output_names: list[str],
    path: Path,
    settings: dict,
    dynamic_shapes,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        exported = torch.export.export(
            module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with patch_trt_interpreter_output_names(output_names):
        engine = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported,
            inputs=input_specs,
            **settings,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    path.write_bytes(engine)
    return elapsed


def _vision_export(model, output_root: Path, device: torch.device) -> float:
    visual = model.vlm.model.visual.to(device=device, dtype=torch.float16).eval()
    module = EdgeRuntimeVision(visual).to(device).eval()
    total_tokens = 640
    sample_inputs = (
        torch.zeros(total_tokens, 1536, device=device, dtype=torch.float16),
        torch.zeros(total_tokens, 36, device=device, dtype=torch.float32),
        torch.tensor([0, total_tokens], device=device, dtype=torch.int32),
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.zeros(4, total_tokens, device=device, dtype=torch.int64),
        torch.zeros(4, total_tokens, device=device, dtype=torch.float16),
    )
    names = list(VISION_PROFILES)
    dtypes = [
        torch.float16,
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int64,
        torch.float16,
    ]
    specs = [
        _make_profiled_input(name, dtype, [VISION_PROFILES[name]])
        for name, dtype in zip(names, dtypes)
    ]
    # Patch merger consumes groups of spatial_merge_size**2 == 4 tokens, so
    # express the divisibility constraint instead of declaring every T valid.
    merged_tokens = torch.export.Dim("vision_merged_tokens", min=160, max=18432)
    tokens = 4 * merged_tokens
    dynamic_shapes = (
        {0: tokens},
        {0: tokens},
        {0: torch.export.Dim("vision_batch_p1", min=1, max=116)},
        # The custom plugin fake does not inspect this carrier's length, so
        # torch.export specializes the sample shape. The TRT Input profile below
        # still declares the C++ runtime's independent [1, 384, 768] range.
        {},
        {1: tokens},
        {1: tokens},
    )
    elapsed = _export_engine(
        module,
        sample_inputs,
        specs,
        ["output", "deepstack_features_0", "deepstack_features_1", "deepstack_features_2"],
        output_root / "visual" / "visual.engine",
        TRT_SETTINGS,
        dynamic_shapes,
    )
    visual.cpu()
    free_cuda_memory(module, sample_inputs)
    return elapsed


def _lm_export(model, output_root: Path, device: torch.device) -> float:
    language = model.vlm.model.language_model.to(device=device, dtype=torch.float16).eval()
    lm_head = model.vlm.lm_head.to(device=device, dtype=torch.float16).eval()
    decoder = getattr(language, "model", language)
    num_layers = len(decoder.layers)
    num_deepstack = 3
    module = make_edge_runtime_lm(decoder, lm_head, num_layers, num_deepstack).eval()

    hidden_size = int(language.config.hidden_size)
    num_kv_heads = int(language.config.num_key_value_heads)
    head_dim = int(language.config.head_dim)
    seq_len = 2
    capacity = 4096
    inputs_embeds = torch.zeros(1, seq_len, hidden_size, device=device, dtype=torch.float16)
    caches = tuple(
        torch.zeros(
            1, 2, num_kv_heads, capacity, head_dim, device=device, dtype=torch.float16
        )
        for _ in range(num_layers)
    )
    rope = torch.zeros(1, capacity, head_dim, device=device, dtype=torch.float32)
    context_lengths = torch.full((1,), seq_len, device=device, dtype=torch.int32)
    cache_start = torch.zeros(1, device=device, dtype=torch.int32)
    last_token_ids = torch.full((1, 1), seq_len - 1, device=device, dtype=torch.int64)
    deepstack = tuple(
        torch.zeros(1, seq_len, hidden_size, device=device, dtype=torch.float16)
        for _ in range(num_deepstack)
    )
    sample_inputs = (
        inputs_embeds,
        *caches,
        rope,
        context_lengths,
        cache_start,
        last_token_ids,
        *deepstack,
    )
    input_names = (
        ["inputs_embeds"]
        + [f"past_key_values_{i}" for i in range(num_layers)]
        + [
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
        ]
        + [f"deepstack_embeds_{i}" for i in range(num_deepstack)]
    )
    prefill_seq = ((1, 1, hidden_size), (1, 1712, hidden_size), (1, 3424, hidden_size))
    decode_seq = ((1, 1, hidden_size),) * 3
    specs = [_make_profiled_input("inputs_embeds", torch.float16, [prefill_seq, decode_seq])]
    cache_profile = ((1, 2, num_kv_heads, capacity, head_dim),) * 3
    for idx in range(num_layers):
        specs.append(
            _make_profiled_input(
                f"past_key_values_{idx}",
                torch.float16,
                [cache_profile, cache_profile],
            )
        )
    fixed_rope = ((1, capacity, head_dim),) * 3
    fixed_one = ((1,),) * 3
    fixed_last = ((1, 1),) * 3
    specs.append(
        _make_profiled_input(
            "rope_rotary_cos_sin", torch.float32, [fixed_rope, fixed_rope]
        )
    )
    specs.append(
        _make_profiled_input("context_lengths", torch.int32, [fixed_one, fixed_one])
    )
    cache_start_spec = _make_profiled_input(
        "kvcache_start_index", torch.int32, [fixed_one, fixed_one]
    )
    _allow_empty_profile_min(cache_start_spec)
    specs.append(cache_start_spec)
    specs.append(
        _make_profiled_input("last_token_ids", torch.int64, [fixed_last, fixed_last])
    )
    for idx in range(num_deepstack):
        specs.append(
            _make_profiled_input(
                f"deepstack_embeds_{idx}",
                torch.float16,
                [prefill_seq, decode_seq],
            )
        )

    seq = torch.export.Dim("lm_seq", min=1, max=3424)
    dynamic_shapes = (
        {1: seq},
        *({} for _ in range(num_layers)),
        {},
        {},
        {},
        {},
        *({1: seq} for _ in range(num_deepstack)),
    )
    patched = patch_language_attention(
        decoder,
        hidden_size=hidden_size,
        num_attention_heads=int(language.config.num_attention_heads),
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        context_attention_mask_type=ContextAttentionMaskType.CAUSAL,
        name="alpamayo-edge-runtime-lm",
    )
    try:
        elapsed = _export_engine(
            module,
            sample_inputs,
            specs,
            ["logits"] + [f"present_key_values_{i}" for i in range(num_layers)],
            output_root / "llm" / "llm.engine",
            LANGUAGE_TRT_SETTINGS,
            dynamic_shapes,
        )
    finally:
        restore_attention(patched)
    language.cpu()
    lm_head.cpu()
    free_cuda_memory(module, sample_inputs, caches, deepstack)
    return elapsed


def _action_export(model, output_root: Path, device: torch.device) -> float:
    model.expert = model.expert.to(device=device, dtype=torch.float16).eval()
    model.action_in_proj = model.action_in_proj.to(device=device, dtype=torch.float16).eval()
    model.action_out_proj = model.action_out_proj.to(device=device, dtype=torch.float16).eval()
    decoder = getattr(model.expert, "model", model.expert)
    num_layers = len(decoder.layers)
    module = make_edge_runtime_action(model, decoder, num_layers).to(device).eval()

    num_kv_heads = 8
    head_dim = 128
    capacity = 4096
    waypoints = 64
    leading = (
        torch.randn(1, waypoints, 2, device=device, dtype=torch.float32),
        torch.zeros(1, device=device, dtype=torch.float32),
        torch.full((1,), 0.1, device=device, dtype=torch.float32),
        torch.full((1,), 3011, device=device, dtype=torch.int32),
        torch.zeros(1, waypoints, head_dim, device=device, dtype=torch.float32),
        torch.arange(waypoints, device=device, dtype=torch.int32).unsqueeze(0),
    )
    k_caches = tuple(
        torch.zeros(1, num_kv_heads, capacity, head_dim, device=device, dtype=torch.float16)
        for _ in range(num_layers)
    )
    v_caches = tuple(torch.zeros_like(value) for value in k_caches)
    sample_inputs = (*leading, *k_caches, *v_caches)
    input_names = (
        [
            "noise_trajectory",
            "time_steps_t0",
            "time_steps_t1",
            "kvcache_start_index",
            "rope_rotary_cos_sin",
            "attention_pos_id",
        ]
        + [f"k_cache_{i}" for i in range(num_layers)]
        + [f"v_cache_{i}" for i in range(num_layers)]
    )
    specs = [
        torch_tensorrt.Input(shape=tuple(value.shape), dtype=value.dtype, name=name)
        for name, value in zip(input_names, sample_inputs)
    ]
    elapsed = _export_engine(
        module,
        sample_inputs,
        specs,
        ["denoised_trajectory"]
        + [f"present_k_cache_{i}" for i in range(num_layers)]
        + [f"present_v_cache_{i}" for i in range(num_layers)],
        output_root / "action" / "action.engine",
        {**TRT_SETTINGS, "offload_module_to_cpu": True},
        tuple({} for _ in sample_inputs),
    )
    free_cuda_memory(module, sample_inputs, k_caches, v_caches)
    return elapsed


def _install_runtime_files(source_root: Path, output_root: Path) -> None:
    for component in ("visual", "llm", "action"):
        source = source_root / component / "config.json"
        target = output_root / component / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in (
        "embedding.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processed_chat_template.json",
    ):
        shutil.copy2(source_root / "llm" / name, output_root / "llm" / name)
    shutil.copy2(
        source_root / "visual" / "preprocessor_config.json",
        output_root / "visual" / "preprocessor_config.json",
    )


def _engine_bindings(path: Path) -> list[str]:
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {path}")
    return [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]


def _validate_abi(output_root: Path) -> None:
    expected_counts = {"visual": 10, "llm": 81, "action": 151}
    expected_required = {
        "visual": {"input", "output", "deepstack_features_2"},
        "llm": {
            "inputs_embeds",
            "past_key_values_35",
            "deepstack_embeds_2",
            "logits",
            "present_key_values_35",
        },
        "action": {
            "noise_trajectory",
            "k_cache_35",
            "v_cache_35",
            "denoised_trajectory",
            "present_k_cache_35",
            "present_v_cache_35",
        },
    }
    filenames = {"visual": "visual.engine", "llm": "llm.engine", "action": "action.engine"}
    for component, filename in filenames.items():
        names = _engine_bindings(output_root / component / filename)
        if len(names) != expected_counts[component]:
            raise RuntimeError(
                f"{component} ABI has {len(names)} bindings; expected "
                f"{expected_counts[component]}: {names}"
            )
        missing = expected_required[component] - set(names)
        if missing:
            raise RuntimeError(f"{component} ABI is missing bindings: {sorted(missing)}")
        print(f"{component} ABI: {len(names)} bindings OK")


def _load_model(model_path: str, device: torch.device):
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Run with the Alpamayo/lerobot Python environment and set PYTHONPATH "
            "to alpamayo/src and Test."
        ) from exc
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.float16).eval()
    model.to(device)
    force_hf_attention(model.vlm.model.visual, "eager", use_cache=False)
    force_hf_attention(model.vlm.model.language_model, "eager")
    force_hf_attention(model.expert, "eager")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export Alpamayo directly with Torch-TensorRT using the exact engine "
            "ABI consumed by TensorRT-Edge-LLM action_inference, then run it."
        )
    )
    parser.add_argument(
        "--model-path",
        default=str(Path.home() / "tensorrt-edgellm-workspace/Alpamayo-R1-10B"),
    )
    parser.add_argument(
        "--runtime-artifacts",
        default=str(
            Path.home()
            / "tensorrt-edgellm-workspace/Alpamayo-R1-10B/engines"
        ),
        help="Validated Edge engine tree used only for config/tokenizer sidecars.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path.home()
            / "tensorrt-edgellm-workspace/Alpamayo-R1-10B/torch_trt_edge_engines"
        ),
    )
    parser.add_argument(
        "--edge-llm-root",
        default="/home/micwilliams/workspace/TensorRT-Edge-LLM",
    )
    parser.add_argument(
        "--input-file",
        default=str(
            Path.home() / "tensorrt-edgellm-workspace/alpamayo_sample/input_action.json"
        ),
    )
    parser.add_argument("--output-file", default="")
    parser.add_argument(
        "--components",
        default="vision,language,action",
        help="Comma-separated engine components to rebuild.",
    )
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output_root = Path(args.output_dir).resolve()
    runtime_root = Path(args.runtime_artifacts).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    components = {item.strip() for item in args.components.split(",") if item.strip()}
    unknown = components - {"vision", "language", "action"}
    if unknown:
        raise ValueError(f"Unknown --components: {sorted(unknown)}")

    load_plugins_for_trt()
    model = _load_model(args.model_path, device)

    timings: dict[str, float] = {}
    if "vision" in components:
        print("Exporting Edge-runtime-compatible visual.engine")
        timings["vision_s"] = _vision_export(model, output_root, device)
        print(f"vision Torch-TRT engine compile: {timings['vision_s']:.3f} s")
    if "language" in components:
        print("Exporting Edge-runtime-compatible llm.engine")
        timings["language_s"] = _lm_export(model, output_root, device)
        print(f"language Torch-TRT engine compile: {timings['language_s']:.3f} s")
    if "action" in components:
        print("Exporting Edge-runtime-compatible action.engine")
        timings["action_s"] = _action_export(model, output_root, device)
        print(f"action Torch-TRT engine compile: {timings['action_s']:.3f} s")
    timings["total_s"] = sum(timings.values())
    print(f"selected Torch-TRT engine compile: {timings['total_s']:.3f} s")

    model.cpu()
    del model
    free_cuda_memory()
    _install_runtime_files(runtime_root, output_root)
    _validate_abi(output_root)
    (output_root / "torch_trt_compile_times.json").write_text(
        json.dumps(timings, indent=2)
        + "\n"
    )

    if args.skip_run:
        return 0

    edge_root = Path(args.edge_llm_root).resolve()
    executable = edge_root / "build/examples/multimodal/action_inference"
    output_file = (
        Path(args.output_file)
        if args.output_file
        else output_root / "action_inference_output.json"
    )
    profile_file = output_root / "action_inference_profile.json"
    command = [
        str(executable),
        "--engineDir",
        str(output_root / "llm"),
        "--multimodalEngineDir",
        str(output_root),
        "--inputFile",
        str(Path(args.input_file).resolve()),
        "--outputFile",
        str(output_file),
        "--dumpProfile",
        "--profileOutputFile",
        str(profile_file),
        "--warmup=3",
        "--noiseSeed=42",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=edge_root, check=True)
    print(f"action_inference output: {output_file}")
    print(f"action_inference profile: {profile_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
