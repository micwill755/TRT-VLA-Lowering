from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_TEST_ROOT = _SCRIPT_DIR.parent
for _p in (_TEST_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mamba_stub import apply as apply_mamba_stub

from trt.plugin.plugin_utils import load_plugins_for_trt, patch_nemotron_mixers
from trt.plugin.attention import PluginAttention, PluginNemotronAttention
from trt.plugin.mamba import PluginNemotronMamba
from trt.plugin.moe import PluginNemotronMoE
from trt.modules.export.language import gather_last_token_hidden
from trt.rope import make_rope_rotary_cos_sin
from trt.measure import parity
from trt.compile import make_input_spec

import torch_tensorrt

TRT_SETTINGS = {
    "enabled_precisions": {torch.float16},
    "min_block_size": 1,
    "use_explicit_typing": True,
    "disable_tf32": True,
}

_WARMUP = 5
_ITERS = 100


def _speedup(eager_ms: float, trt_ms: float) -> str:
    if eager_ms <= 0.0 or trt_ms <= 0.0:
        return "n/a (benchmark skipped)"
    return f"{eager_ms / trt_ms:.3f}x"


def _cuda_time_ms(fn, device, warmup: int = _WARMUP, iters: int = _ITERS) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / iters


def _decoder(model):
    return getattr(model, "backbone", None) or model.model

def _kind(mixer) -> str:
    name = type(mixer).__name__
    if "Mamba" in name:
        return "mamba"
    if "Attention" in name:
        return "attention"
    if "MoE" in name or "Moe" in name:
        return "moe"
    return "mlp"

class NemotronExportModule(nn.Module):
    """Hybrid decoder: plugin attention / mamba / moe, native MLP.
    Flat I/O matches Edge-LLM NemotronH (linear KV, not paged)::
        inputs_embeds, rope, ctx_len, kv_start, last_token_ids,
        *kv[Na], *conv[Nm], *ssm[Nm]
            -> logits, *present_kv, *present_conv, *present_ssm
    """
    def __init__(self, model):
        super().__init__()
        decoder = _decoder(model)
        self.layers = decoder.layers
        self.norm = decoder.norm_f
        self.lm_head = model.lm_head
        self.kinds = [_kind(block.mixer) for block in self.layers]
        self.num_attn = self.kinds.count("attention")
        self.num_mamba = self.kinds.count("mamba")
    
    def forward(
        self,
        inputs_embeds,
        rope_rotary_cos_sin,
        context_lengths,
        kvcache_start_index,
        last_token_ids,
        *states,
    ):
        Na, Nm = self.num_attn, self.num_mamba
        kvs = list(states[:Na])
        convs = list(states[Na : Na + Nm])
        ssms = list(states[Na + Nm : Na + 2 * Nm])
        kv_i = conv_i = 0
        hidden = inputs_embeds
        present_kv, present_conv, present_ssm = [], [], []
        for block, kind in zip(self.layers, self.kinds):
            residual = hidden
            hidden = block.norm(hidden)
            mixer = block.mixer
            if kind == "attention":
                # PluginNemotronAttention → torch.ops.trt.attention_plugin
                hidden, kv = mixer(
                    hidden_states=hidden,
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    past_key_value=kvs[kv_i],
                    ctx_len=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                )
                present_kv.append(kv)
                kv_i += 1
            elif kind == "mamba":
                # PluginNemotronMamba → causal_conv1d + update_ssm_state
                hidden, conv_out, ssm_out = mixer(
                    hidden, convs[conv_i], ssms[conv_i], context_lengths
                )
                present_conv.append(conv_out)
                present_ssm.append(ssm_out)
                conv_i += 1
            else:
                # MLP (native TRT GEMM) or PluginNemotronMoE → nvfp4_moe_plugin
                hidden = mixer(hidden)
            hidden = residual + hidden
        hidden = self.norm(hidden)
        last = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last).float()
        return (logits, *present_kv, *present_conv, *present_ssm)

def allocate_plugin_states(model, config, batch, max_seq_len, device, dtype):
    kinds = [_kind(block.mixer) for block in _decoder(model).layers]
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // config.num_attention_heads)
    conv_dim = (
        int(config.mamba_num_heads) * int(config.mamba_head_dim)
        + 2 * int(config.n_groups) * int(config.ssm_state_size)
    )
    conv_kernel = int(getattr(config, "conv_kernel", 4))
    kvs, convs, ssms = [], [], []
    for kind in kinds:
        if kind == "attention":
            kvs.append(
                torch.zeros(
                    batch, 2, int(config.num_key_value_heads), max_seq_len, head_dim,
                    device=device, dtype=dtype,
                )
            )
        elif kind == "mamba":
            convs.append(
                torch.zeros(batch, conv_dim, conv_kernel, device=device, dtype=dtype)
            )
            ssms.append(
                torch.zeros(
                    batch,
                    int(config.mamba_num_heads),
                    int(config.mamba_head_dim),
                    int(config.ssm_state_size),
                    device=device,
                    dtype=dtype,
                )
            )
    return kvs, convs, ssms

def load_config(device, checkpoint):
    apply_mamba_stub()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device=device, dtype=torch.float16).eval()
    return model.config, model

def load_tokenizer(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def encode_prompt(tokenizer, prompt: str, device):
    encoded = tokenizer(prompt, return_tensors="pt")
    return (
        encoded["input_ids"].to(device=device),
        encoded["attention_mask"].to(device=device),
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    checkpoint = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    max_seq_len = 128

    load_plugins_for_trt()
    config, model = load_config(device, checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    input_ids, attention_mask = encode_prompt(tokenizer, "Hello.", device)
    embeddings = model.get_input_embeddings()(input_ids)
    bsz, prompt_len, hidden = embeddings.shape
    # Time/export the engine capacity, not the 2-token prompt. HF naive Mamba
    # still pads internally to chunk_size (256 on 4B); TRT scans context_lengths.
    if prompt_len < max_seq_len:
        pad = max_seq_len - prompt_len
        embeddings = torch.cat(
            [embeddings, torch.zeros(bsz, pad, hidden, device=device, dtype=embeddings.dtype)],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(bsz, pad, device=device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
    seq_len = int(embeddings.shape[1])
    chunk_size = int(getattr(config, "chunk_size", 0) or 0)

    with torch.no_grad():
        eager = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        eager_logits = eager.logits[:, -1, :]
    eager_elapsed_ms = _cuda_time_ms(
        lambda: model(inputs_embeds=embeddings, attention_mask=attention_mask),
        device,
    )

    patch_nemotron_mixers(model, config)
    for block in _decoder(model).layers:
        if isinstance(block.mixer, PluginNemotronMoE):
            block.mixer.prepare_for_export()

    lm = NemotronExportModule(model).eval().to(device)
    
    # Nemotron-H attention is NoPE (no rope_theta / rotary_emb); this is identity cos/sin.
    rope = make_rope_rotary_cos_sin(config, max_seq_len, device, language_model=_decoder(model))
    ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.int64)
    kv_start = torch.empty(0, dtype=torch.int32, device=device)
    kvs, convs, ssms = allocate_plugin_states(
        model, config, bsz, max_seq_len, device, dtype
    )
    flat = (embeddings, rope, ctx_len, kv_start, last_token_ids, *kvs, *convs, *ssms)
    
    with torch.no_grad():
        plugin_out = lm(*flat)
        plugin_logits = plugin_out[0]
    
    exported = torch.export.export(lm, args=flat, strict=False)
    trt_engine = torch_tensorrt.dynamo.compile(
        exported, inputs=make_input_spec(flat), **TRT_SETTINGS
    )

    with torch.no_grad():
        trt_out = trt_engine(*flat)
        trt_logits = trt_out[0]
    trt_elapsed_ms = _cuda_time_ms(lambda: trt_engine(*flat), device)

    # attention_plugin / causal_conv1d / update_ssm_state return zeros in eager
    # PyTorch (Dynamo shape stubs). MLP still runs, so this is not all-zero, but
    # it is not a numerical check. PI05 only reports unpatched eager vs TRT.
    parity("nemotron wrapper (stub ops)", eager_logits, plugin_logits)
    parity("nemotron stub vs TRT", plugin_logits, trt_logits)
    parity("nemotron eager vs TRT", eager_logits, trt_logits)

    print(
        f"seq_len={seq_len}  hf_chunk_size={chunk_size}  "
        f"(HF naive Mamba pads prefill to chunk_size; no mamba-ssm CUDA kernels)"
    )
    print(f"lm eager execute: {eager_elapsed_ms:.3f} ms")
    print(f"lm trt execute: {trt_elapsed_ms:.3f} ms")
    print(f"lm speedup: {_speedup(eager_elapsed_ms, trt_elapsed_ms)}")

if __name__ == "__main__":
    main()
