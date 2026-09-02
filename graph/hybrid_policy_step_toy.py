"""Screenshot graph via EdgeExporter.export_for_policy() — no checkpoint.

  graph 1  vision (conv/bn/relu) | edge::fuse_prefix | language (sdpa)
  graph 2  execute_engine | fuse_prefix | execute_engine

Adapters are identity: FakePolicy already has vision_tower + language_model.
export_for_policy still composes one PolicyStep and compiles it once.

Run::

    python graph/hybrid_policy_step_toy.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_GRAPH_DIR = Path(__file__).resolve().parent
if str(_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt

from exporter import (
    EdgeConfig,
    EdgeExporter,
    PolicyStep,
    dump_graph,
    named_runtime_preview,
)

logging.getLogger("torch._library.fake_class_registry").setLevel(logging.ERROR)

HIDDEN = 16
HEADS = 4


class TinyVision(nn.Module):
    """conv → bn → relu → tokens [B, HW, C]."""

    def __init__(self, channels: int = HIDDEN):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(pixel_values)))
        return x.flatten(2).transpose(1, 2)


class TinyLanguage(nn.Module):
    """4D BHND SDPA on the fused prefix."""

    def __init__(self, hidden: int = HIDDEN, num_heads: int = HEADS):
        super().__init__()
        if hidden % num_heads:
            raise ValueError("hidden must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = prefix.shape
        qkv = prefix.reshape(batch, seq, self.num_heads, self.head_dim)
        qkv = qkv.transpose(1, 2).contiguous()
        return F.scaled_dot_product_attention(qkv, qkv, qkv)


class FakePolicy(nn.Module):
    """Bag of submodules. export_for_policy discovers these names."""

    def __init__(self):
        super().__init__()
        self.vision_tower = TinyVision()
        self.language_model = TinyLanguage()


def main() -> None:
    device = torch.device("cuda")
    policy = FakePolicy().to(device).eval()
    pixels = torch.randn(1, 3, 8, 8, device=device)
    lang_embeds = torch.randn(1, 8, HIDDEN, device=device)
    inputs = {"pixel_values": pixels, "lang_embeds": lang_embeds}

    step = PolicyStep(policy.vision_tower, policy.language_model).eval()
    graph1 = torch.export.export(step, (pixels, lang_embeds), strict=False)
    dump_graph(graph1.graph_module, "graph 1  PolicyStep (vision | fuse_prefix | language)")

    exporter = EdgeExporter()
    compiled = exporter.export_for_policy(
        policy,
        inputs,
        config=EdgeConfig(decompose_attention=False),
    )
    dump_graph(
        compiled,
        "graph 2a  dynamo.compile (engine modules + leftover fuse_prefix)",
    )
    leftover = getattr(compiled, "_run_on_gpu_1", None)
    if leftover is not None:
        dump_graph(leftover, "graph 2a leftover  _run_on_gpu_1")

    retraced = torch_tensorrt.dynamo.export(
        compiled, arg_inputs=(pixels, lang_embeds)
    )
    dump_graph(retraced.graph_module, "graph 2b  retrace (screenshot IR)")
    named_runtime_preview(retraced.graph_module)

    with torch.no_grad():
        eager = step(pixels, lang_embeds)
        trt = compiled(pixels, lang_embeds)
    print("\nmax abs err:", (eager - trt).abs().max().item())

    out = Path("/tmp/hybrid_policy_step.pt2")
    torch_tensorrt.save(
        compiled,
        file_path=str(out),
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=(pixels, lang_embeds),
    )
    print(f"AOTI package: {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
