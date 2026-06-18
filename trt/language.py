"""LLM export/runtime aligned with LLMEngineRunner I/O."""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from trt.attention import PluginAttention
from trt.compile import compile_trt_module, save_trt_engine_module
from trt.plugin_utils import set_plugin_config_from_model


def _as_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def language_head_dim(config: Any) -> int:
    return int(
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
    )


def install_plugin_attention(
    lm: nn.Module,
    config: Any,
    rope_cache: torch.Tensor,
    *,
    enable_bidirectional_prefill: int = 1,
) -> None:
    for i, layer in enumerate(lm.layers):
        layer.self_attn = PluginAttention(
            layer.self_attn,
            config,
            layer_idx=i,
            rope_cache=rope_cache,
            enable_bidirectional_prefill=enable_bidirectional_prefill,
        ).eval()


def build_rope_cache(
    lm: nn.Module,
    config: Any,
    *,
    max_seq_len: int,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if position_ids is None:
        position_ids = torch.arange(max_seq_len, device=device).view(1, max_seq_len)
    position_ids = position_ids.to(device=device)[:, :max_seq_len]

    head_dim = language_head_dim(config)
    with torch.no_grad():
        dummy = torch.ones(
            position_ids.shape[0],
            max_seq_len,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        cos, sin = lm.rotary_emb(dummy, position_ids)
        h2 = head_dim // 2
        return torch.cat(
            [cos[:, :max_seq_len, :h2].float(), sin[:, :max_seq_len, :h2].float()],
            dim=-1,
        )


def runner_rope_input(
    rope_cache: torch.Tensor,
    *,
    batch_size: int,
    max_kv_capacity: int,
) -> torch.Tensor:
    rope = rope_cache[:, :max_kv_capacity, :].to(dtype=torch.float32)
    if rope.shape[0] == 1 and batch_size > 1:
        rope = rope.expand(batch_size, -1, -1)
    return rope.contiguous()


def empty_kv_caches(
    *,
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    max_kv_capacity: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros(
            batch_size,
            2,
            num_kv_heads,
            max_kv_capacity,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_layers)
    )


def prefill_inputs(
    inputs_embeds: torch.Tensor,
    config: Any,
    *,
    max_kv_capacity: int,
    rope_cache: torch.Tensor,
) -> dict[str, Any]:
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device

    past_key_values = empty_kv_caches(
        batch_size=batch_size,
        num_layers=int(config.num_hidden_layers),
        num_kv_heads=int(config.num_key_value_heads),
        max_kv_capacity=max_kv_capacity,
        head_dim=language_head_dim(config),
        device=device,
        dtype=inputs_embeds.dtype,
    )

    return {
        "inputs_embeds": inputs_embeds,
        "past_key_values": past_key_values,
        "rope_rotary_cos_sin": runner_rope_input(
            rope_cache,
            batch_size=batch_size,
            max_kv_capacity=max_kv_capacity,
        ),
        "context_lengths": torch.full((batch_size,), seq_len, dtype=torch.int32, device=device),
        "last_token_ids": torch.full((batch_size, 1), seq_len - 1, dtype=torch.int64, device=device),
        "kvcache_start_index": torch.empty(0, dtype=torch.int32, device=device),
    }


def stack_prefix_kv_from_present(
    present_key_values: Sequence[torch.Tensor],
    *,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_k = torch.stack([kv[:, 0, :, :seq_len, :] for kv in present_key_values], dim=0)
    prefix_v = torch.stack([kv[:, 1, :, :seq_len, :] for kv in present_key_values], dim=0)
    return prefix_k, prefix_v


@torch.no_grad()
def run_prefix_language_eager(
    language_model: nn.Module,
    prefix_embs: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lm_dtype = next(language_model.parameters()).dtype
    prefix_embs = prefix_embs.to(dtype=lm_dtype)
    out = language_model(
        inputs_embeds=prefix_embs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
    )
    cache = out.past_key_values
    prefix_k = torch.stack([layer.keys for layer in cache.layers], dim=0)
    prefix_v = torch.stack([layer.values for layer in cache.layers], dim=0)
    return out.last_hidden_state, prefix_k, prefix_v


class VLAPluginCausalLM(nn.Module):
    """Plugin-attention causal LM with LLMEngineRunner-compatible forward I/O."""

    def __init__(
        self,
        lm: nn.Module,
        lm_head: nn.Module | None,
        config: Any,
        *,
        rope_cache: torch.Tensor,
        enable_bidirectional_prefill: int = 1,
        export_hidden_states: bool = False,
        num_ds: int = 0,
    ):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.config = config
        self.enable_bidirectional_prefill = int(enable_bidirectional_prefill)
        self.export_hidden_states = bool(export_hidden_states)
        self.num_ds = int(num_ds)
        self.register_buffer("rope_cache", rope_cache)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ds_stack: torch.Tensor | None = None,
    ):
        del rope_rotary_cos_sin, position_ids, attention_mask

        hidden = inputs_embeds
        present_key_values = []
        seq_len = hidden.shape[1]

        for layer_idx, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, present_kv = layer.self_attn(
                hidden_states=hidden,
                past_key_value=past_key_values[layer_idx],
                ctx_len=context_lengths,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = residual + _as_tensor(hidden)

            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = _as_tensor(layer.mlp(hidden))
            hidden = residual + hidden

            if ds_stack is not None and self.num_ds > 0 and layer_idx < self.num_ds:
                hidden = hidden + ds_stack[layer_idx, :, :seq_len, :]

            present_key_values.append(present_kv)

        hidden = _as_tensor(self.lm.norm(hidden))

        logits = None
        if self.lm_head is not None:
            if last_token_ids.ndim == 1:
                index = last_token_ids.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
            else:
                index = last_token_ids.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            gathered = hidden.gather(1, index.to(dtype=torch.long))
            logits = self.lm_head(gathered.squeeze(1)).to(torch.float32)

        if self.export_hidden_states:
            return logits, hidden, tuple(present_key_values)
        return logits, tuple(present_key_values)


class FlatLLMRunnerExportWrapper(nn.Module):
    """Flatten per-layer KV tensors for torch.export / TRT serialization."""

    def __init__(self, model: VLAPluginCausalLM):
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        *past_key_values: torch.Tensor,
        ds_stack: torch.Tensor | None = None,
    ):
        outputs = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            last_token_ids,
            kvcache_start_index,
            ds_stack=ds_stack,
        )
        if self.model.export_hidden_states:
            logits, hidden_states, present_kvs = outputs
            if self.model.lm_head is None:
                return (hidden_states, *present_kvs)
            return (logits, hidden_states, *present_kvs)
        logits, present_kvs = outputs
        return (logits, *present_kvs)


def llm_runner_past_key_value_names(num_layers: int, *, is_past: bool = True) -> list[str]:
    prefix = "past_key_values" if is_past else "present_key_values"
    return [f"{prefix}_{i}" for i in range(num_layers)]


def llm_runner_input_names(num_layers: int) -> list[str]:
    return [
        "inputs_embeds",
        *llm_runner_past_key_value_names(num_layers, is_past=True),
        "rope_rotary_cos_sin",
        "context_lengths",
        "last_token_ids",
        "kvcache_start_index",
    ]


def llm_runner_output_names(num_layers: int, *, include_hidden_states: bool = False, include_logits: bool = True) -> list[str]:
    names: list[str] = []
    if include_logits:
        names.append("logits")
    if include_hidden_states:
        names.append("hidden_states")
    names.extend(llm_runner_past_key_value_names(num_layers, is_past=False))
    return names


def make_vla_plugin_causal_lm(
    lm: nn.Module,
    lm_head: nn.Module | None,
    config: Any,
    *,
    max_seq_len: int,
    max_kv_capacity: int,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
    enable_bidirectional_prefill: int = 1,
    export_hidden_states: bool = False,
    num_ds: int = 0,
) -> VLAPluginCausalLM:
    lm = copy.deepcopy(lm).to(device=device, dtype=torch.float16).eval()
    if lm_head is not None:
        lm_head = copy.deepcopy(lm_head).to(device=device, dtype=torch.float16).eval()

    rope_cache = build_rope_cache(
        lm,
        config,
        max_seq_len=max_seq_len,
        device=device,
        position_ids=position_ids,
    )

    set_plugin_config_from_model(config, max_kv_capacity)
    install_plugin_attention(
        lm,
        config,
        rope_cache,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
    )

    return VLAPluginCausalLM(
        lm,
        lm_head,
        config,
        rope_cache=rope_cache,
        enable_bidirectional_prefill=enable_bidirectional_prefill,
        export_hidden_states=export_hidden_states,
        num_ds=num_ds,
    ).eval()


def _llm_runner_sample_args(inputs: dict[str, Any]) -> tuple[Any, ...]:
    return (
        inputs["inputs_embeds"],
        inputs["rope_rotary_cos_sin"],
        inputs["context_lengths"],
        inputs["last_token_ids"],
        inputs["kvcache_start_index"],
        *inputs["past_key_values"],
    )


@torch.no_grad()
def run_vla_lm_prefill(
    model: VLAPluginCausalLM | nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    max_kv_capacity: int,
    rope_cache: torch.Tensor | None = None,
    config: Any | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[torch.Tensor, ...]]:
    if config is None:
        config = model.config
    if rope_cache is None:
        rope_cache = model.rope_cache

    inputs = prefill_inputs(
        inputs_embeds,
        config,
        max_kv_capacity=max_kv_capacity,
        rope_cache=rope_cache,
    )
    outputs = model(
        inputs["inputs_embeds"],
        inputs["past_key_values"],
        inputs["rope_rotary_cos_sin"],
        inputs["context_lengths"],
        inputs["last_token_ids"],
        inputs["kvcache_start_index"],
    )
    if getattr(model, "export_hidden_states", False):
        logits, hidden, present_kvs = outputs
        return logits, hidden, present_kvs
    logits, present_kvs = outputs
    return logits, None, present_kvs


def compile_vla_lm_trt(
    model: VLAPluginCausalLM,
    inputs_embeds: torch.Tensor,
    *,
    max_kv_capacity: int,
    device: torch.device,
    settings: dict[str, Any],
) -> nn.Module:
    wrapper = FlatLLMRunnerExportWrapper(model).eval()
    inputs = prefill_inputs(
        inputs_embeds.to(device=device, dtype=torch.float16),
        model.config,
        max_kv_capacity=max_kv_capacity,
        rope_cache=model.rope_cache,
    )
    return compile_trt_module(wrapper, _llm_runner_sample_args(inputs), settings)


class PI05PrefillLanguageAdapter:
    """Adapt compiled LLM runner TRT module to PI0.5 prefix_k/v prefill API."""

    def __init__(self, trt_lm: nn.Module, config: Any, *, max_kv_capacity: int, rope_cache: torch.Tensor):
        self.trt_lm = trt_lm
        self.config = config
        self.max_kv_capacity = int(max_kv_capacity)
        self.rope_cache = rope_cache

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: list[torch.Tensor] | None = None,
        ctx_len: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        del kv_caches, ctx_len
        inputs = prefill_inputs(
            inputs_embeds,
            self.config,
            max_kv_capacity=self.max_kv_capacity,
            rope_cache=self.rope_cache,
        )
        outputs = self.trt_lm(*_llm_runner_sample_args(inputs))
        present_kvs = outputs[1:]
        seq_len = int(inputs_embeds.shape[1])
        prefix_k, prefix_v = stack_prefix_kv_from_present(present_kvs, seq_len=seq_len)
        return None, prefix_k, prefix_v


def _build_llm_runner_config(
    config: Any,
    *,
    max_input_len: int,
    max_kv_cache_capacity: int,
    max_batch_size: int,
    model_type: str,
    input_names: list[str],
    output_names: list[str],
    example_inputs: tuple[Any, ...],
    example_output: tuple[Any, ...],
) -> dict[str, Any]:
    head_dim = language_head_dim(config)
    return {
        "model_type": model_type,
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_key_value_heads": int(config.num_key_value_heads),
        "num_attention_heads": int(config.num_attention_heads),
        "head_dim": head_dim,
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "partial_rotary_factor": float(getattr(config, "partial_rotary_factor", 1.0)),
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", max_kv_cache_capacity)
        ),
        "builder_config": {
            "max_batch_size": int(max_batch_size),
            "max_input_len": int(max_input_len),
            "max_kv_cache_capacity": int(max_kv_cache_capacity),
            "max_lora_rank": 0,
            "eagle_base": False,
            "trt_native_ops": False,
        },
        "lm_runtime": "llm_engine_runner",
        "input_names": input_names,
        "output_names": output_names,
        "inputs": {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in zip(input_names, example_inputs)},
        "outputs": [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)} for name, t in zip(output_names, example_output)],
    }


def save_lm_engine(
    model: VLAPluginCausalLM,
    inputs_embeds: torch.Tensor,
    engine_dir: str | pathlib.Path,
    *,
    max_kv_capacity: int | None = None,
    max_batch_size: int = 1,
    model_type: str = "gemma",
    manifest: dict[str, Any] | None = None,
    engine_file: str = "model.engine",
    trt_settings: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Export LLMEngineRunner-aligned TRT engine + config under ``llm/``."""
    engine_dir = pathlib.Path(engine_dir)
    max_kv_capacity = int(max_kv_capacity or inputs_embeds.shape[1])
    cfg = model.config
    num_layers = int(cfg.num_hidden_layers)

    wrapper = FlatLLMRunnerExportWrapper(model).eval()
    inputs = prefill_inputs(
        inputs_embeds.to(device=inputs_embeds.device, dtype=torch.float16).contiguous(),
        cfg,
        max_kv_capacity=max_kv_capacity,
        rope_cache=model.rope_cache,
    )
    sample_args = _llm_runner_sample_args(inputs)
    input_names = llm_runner_input_names(num_layers)
    output_names = llm_runner_output_names(
        num_layers,
        include_hidden_states=model.export_hidden_states,
        include_logits=model.lm_head is not None,
    )

    with torch.no_grad():
        example_output = wrapper(*sample_args)

    save_trt_engine_module(
        wrapper,
        sample_args,
        engine_dir,
        engine_file=engine_file,
        model_type=model_type,
        component="llm",
        input_names=input_names,
        output_names=output_names,
        example_output=example_output,
        extra_config={
            **_build_llm_runner_config(
                cfg,
                max_input_len=int(inputs_embeds.shape[1]),
                max_kv_cache_capacity=max_kv_capacity,
                max_batch_size=max_batch_size,
                model_type=model_type,
                input_names=input_names,
                output_names=output_names,
                example_inputs=sample_args,
                example_output=example_output if isinstance(example_output, tuple) else (example_output,),
            ),
            "max_seq_len": max_kv_capacity,
            "rope_cache_file": "rope_cache.pt",
        },
        trt_settings=trt_settings,
    )

    torch.save(model.rope_cache.detach().cpu(), engine_dir / "rope_cache.pt")

    if manifest is not None:
        manifest_path = engine_dir.parent / "vla_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return engine_dir / engine_file
