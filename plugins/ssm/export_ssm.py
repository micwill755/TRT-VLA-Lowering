"""Export a Nemotron-H-style Mamba mixer (and optional GDN) through plugin converters.

Same flow as ``vla/test_vla_pi05_e2e_one_shot.py``:

  custom op stub → nn.Module wrapper → Dynamo converter → Edge-LLM plugin

Run from the Test repo root::

    export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
    python -m plugins.ssm.export_ssm
    python -m plugins.ssm.export_ssm --example gdn --no-compile
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
from plugins.ssm.modules import PluginGatedDeltaNet, PluginMambaMixer


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_mixer(*, compile_engine: bool) -> None:
    device = _device()
    dtype = torch.float16
    mixer = PluginMambaMixer(hidden_size=256, nheads=4, head_dim=64, dstate=64, ngroups=2)
    mixer = mixer.to(device=device, dtype=dtype).eval()

    batch, seq_len = 1, 8
    hidden = torch.randn(batch, seq_len, mixer.hidden_size, device=device, dtype=dtype)
    conv_state = torch.zeros(batch, mixer.conv_dim, mixer.conv.kernel_size, device=device, dtype=dtype)
    ssm_state = torch.zeros(
        batch, mixer.nheads, mixer.head_dim, mixer.dstate, device=device, dtype=dtype
    )
    context_lengths = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    args = (hidden, conv_state, ssm_state, context_lengths)

    print(
        "SSM mixer plugins: causal_conv1d + update_ssm_state "
        f"(nheads={mixer.nheads}, dim={mixer.head_dim}, dstate={mixer.dstate})"
    )
    export_and_maybe_compile(mixer, args, label="ssm_mixer", compile_engine=compile_engine)


def export_gdn(*, compile_engine: bool) -> None:
    device = _device()
    dtype = torch.float16
    gdn = PluginGatedDeltaNet(num_k_heads=4, num_v_heads=4, head_dim=128)
    gdn = gdn.to(device=device, dtype=dtype).eval()

    batch, seq_len = 1, 8
    q = torch.randn(batch, seq_len, gdn.num_k_heads, gdn.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, gdn.num_v_heads, gdn.head_dim, device=device, dtype=dtype)
    a = torch.randn(batch, seq_len, gdn.num_v_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    h0 = torch.zeros(
        batch, gdn.num_v_heads, gdn.head_dim, gdn.head_dim, device=device, dtype=torch.float32
    )
    context_lengths = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    args = (q, k, v, a, b, h0, context_lengths)

    print("GDN plugin: gated_delta_net (k_dim=v_dim=128)")
    export_and_maybe_compile(gdn, args, label="gated_delta_net", compile_engine=compile_engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=("mixer", "gdn", "all"), default="mixer")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    compile_engine = not args.no_compile

    load_example_plugins(include_attention=False, load_so=compile_engine)
    if args.example in ("mixer", "all"):
        export_mixer(compile_engine=compile_engine)
    if args.example in ("gdn", "all"):
        export_gdn(compile_engine=compile_engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
