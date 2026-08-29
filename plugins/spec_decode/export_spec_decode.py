"""Export speculative-decoding plugin pieces.

Two examples:

* ``dflash`` — ``DFlashTargetKVCacheUpdate`` (paged KV, the DFlash-specific plugin)
* ``tree_attn`` — ``AttentionPlugin`` with ``enable_tree_attention=1`` (same
  converter as VLA language, linear KV cache)

Run from the Test repo root::

    export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
    python -m plugins.spec_decode.export_spec_decode --example dflash
    python -m plugins.spec_decode.export_spec_decode --example tree_attn
    python -m plugins.spec_decode.export_spec_decode --example all --no-compile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parents[2]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import torch

from plugins.common import export_and_maybe_compile, load_example_plugins
from plugins.spec_decode.modules import PluginDFlashKVUpdate, PluginTreeAttention, paged_kv_shape
from plugins.spec_decode.ops import KV_PAGE_SIZE


def export_dflash(*, compile_engine: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    batch, delta_len = 1, 4
    num_kv_heads, head_dim = 2, 64
    pages_per_slot = 2
    rotary_dim = head_dim
    capacity = pages_per_slot * KV_PAGE_SIZE

    module = PluginDFlashKVUpdate(pages_per_slot=pages_per_slot).to(device).eval()
    k_delta = torch.randn(batch, delta_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_delta = torch.randn_like(k_delta)
    past_kv = torch.zeros(
        *paged_kv_shape(batch, pages_per_slot, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    rope = torch.zeros(1, capacity, rotary_dim, device=device, dtype=torch.float32)
    delta_start = torch.zeros(batch, device=device, dtype=torch.int32)
    delta_lengths = torch.full((batch,), delta_len, device=device, dtype=torch.int32)
    args = (k_delta, v_delta, past_kv, rope, delta_start, delta_lengths)

    print(
        "Spec-decode plugin: DFlashTargetKVCacheUpdate "
        f"(pages_per_slot={pages_per_slot}, KV_PAGE_SIZE={KV_PAGE_SIZE})"
    )
    export_and_maybe_compile(module, args, label="dflash_kv_update", compile_engine=compile_engine)


def export_tree_attn(*, compile_engine: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    batch, seq_len, hidden = 1, 8, 256
    num_q_heads, num_kv_heads, head_dim = 4, 2, 64
    capacity = 64
    rotary_dim = head_dim

    module = PluginTreeAttention(hidden, num_q_heads, num_kv_heads, head_dim)
    module = module.to(device=device, dtype=dtype).eval()

    hidden_states = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    past_kv = torch.zeros(batch, 2, num_kv_heads, capacity, head_dim, device=device, dtype=dtype)
    context_lengths = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    rope = torch.zeros(1, capacity, rotary_dim, device=device, dtype=torch.float32)
    kv_start = torch.empty(0, device=device, dtype=torch.int32)
    tree_mask = torch.ones(batch, seq_len, seq_len, device=device, dtype=torch.int32)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).expand(batch, -1)
    args = (hidden_states, past_kv, context_lengths, rope, kv_start, tree_mask, pos_ids)

    print(
        "Spec-decode plugin: AttentionPlugin(enable_tree_attention=1) "
        "(VLA converter ABI: split Q/K/V, linear KV cache)"
    )
    export_and_maybe_compile(module, args, label="tree_attention", compile_engine=compile_engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=("dflash", "tree_attn", "all"), default="dflash")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    compile_engine = not args.no_compile

    load_example_plugins(
        include_attention=args.example in ("tree_attn", "all"),
        load_so=compile_engine,
    )

    if args.example in ("dflash", "all"):
        export_dflash(compile_engine=compile_engine)
    if args.example in ("tree_attn", "all"):
        export_tree_attn(compile_engine=compile_engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
