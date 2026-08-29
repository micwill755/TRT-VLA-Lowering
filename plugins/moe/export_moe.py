"""Export a Qwen3-style FP16 MoE block through ``Fp16MoePlugin``.

The gate GEMM stays a native TRT MatMul. Softmax, top-k, permute, and
grouped expert GEMMs are one plugin node.

Run from the Test repo root::

    export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
    python -m plugins.moe.export_moe
    python -m plugins.moe.export_moe --no-compile
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
from plugins.moe.modules import PluginFp16MoE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    load_example_plugins(include_attention=False, load_so=not args.no_compile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moe = PluginFp16MoE(
        hidden_size=128,
        moe_inter_size=64,
        num_experts=128,
        top_k=2,
    ).to(device=device, dtype=torch.float16).eval()

    batch, seq_len = 1, 8
    hidden = torch.randn(batch, seq_len, moe.hidden_size, device=device, dtype=torch.float16)
    print(
        "MoE plugin: Fp16MoePlugin "
        f"(E={moe.num_experts}, H={moe.hidden_size}, I={moe.moe_inter_size}, top_k={moe.top_k})"
    )
    export_and_maybe_compile(
        moe,
        (hidden,),
        label="fp16_moe",
        compile_engine=not args.no_compile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
