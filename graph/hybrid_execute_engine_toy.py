"""Screenshot graph 1 → 2 via EdgeExporter.export().

  graph 1  aten conv/bn/relu + my_custom::custom_op + sdpa
  graph 2  tensorrt::execute_engine  |  custom_op (AOTI)  |  execute_engine

Run::

    python graph/hybrid_execute_engine_toy.py
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

from exporter import EdgeConfig, EdgeExporter, dump_graph, named_runtime_preview

logging.getLogger("torch._library.fake_class_registry").setLevel(logging.ERROR)

CUSTOM_OP = "my_custom::custom_op"


@torch.library.custom_op(CUSTOM_OP, mutates_args=())
def custom_op(x: torch.Tensor) -> torch.Tensor:
    return x * 1.0


@custom_op.register_fake  # type: ignore[misc]
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class HybridToy(nn.Module):
    """conv → bn → relu → custom_op → attn."""

    def __init__(self, channels: int = 16, num_heads: int = 4):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = torch.ops.my_custom.custom_op(x)
        b, _, h, w = x.shape
        qkv = x.flatten(2).reshape(b, self.num_heads, self.head_dim, h * w)
        qkv = qkv.transpose(2, 3).contiguous()
        return F.scaled_dot_product_attention(qkv, qkv, qkv)


def main() -> None:
    device = torch.device("cuda")
    model = HybridToy().to(device).eval()
    x = torch.randn(1, 3, 8, 8, device=device)

    exporter = EdgeExporter()
    config = EdgeConfig(
        require_full_compilation=False,
        torch_executed_ops={torch.ops.my_custom.custom_op.default},
        decompose_attention=False,
    )
    graph1 = torch.export.export(model, (x,), strict=False)
    dump_graph(graph1.graph_module, "graph 1  torch.export (aten + custom_op)")

    compiled = exporter.export(model, (x,), config=config)
    dump_graph(compiled, "graph 2a  dynamo.compile (engine modules + leftover custom_op)")

    retraced = torch_tensorrt.dynamo.export(compiled, arg_inputs=(x,))
    dump_graph(retraced.graph_module, "graph 2b  retrace (screenshot IR)")
    named_runtime_preview(retraced.graph_module)

    with torch.no_grad():
        eager = model(x)
        trt = compiled(x)
    print("\nmax abs err:", (eager - trt).abs().max().item())

    out = Path("/tmp/hybrid_toy.pt2")
    torch_tensorrt.save(
        compiled,
        file_path=str(out),
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=(x,),
    )
    print(f"AOTI package: {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
