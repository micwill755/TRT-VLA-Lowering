"""Named Edge-LLM runtime ops. Same engine class; behavior is the operator.

Torch-TRT compile still emits ``tensorrt::execute_engine``. Graph 3 rewrites
those nodes to ``edgellm::{vision_tower,text_encoder,action_expert,...}``.
Each op is the same TensorRT engine object plus a call through
``tensorrt::execute_engine`` (later: the matching Edge LLM C API).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch_tensorrt  # noqa: F401  — registers __torch__.torch.classes.tensorrt.Engine

# Match C++ ``tensorrt::execute_engine`` (see torch.ops.tensorrt.execute_engine schema).
_ENGINE_SIG = (
    "(Tensor[] input_tensors, __torch__.torch.classes.tensorrt.Engine engine) -> Tensor[]"
)


def _execute(input_tensors, engine):
    return torch.ops.tensorrt.execute_engine.default(input_tensors, engine)


def _register(name: str):
    @torch.library.custom_op(f"edgellm::{name}", mutates_args=(), schema=_ENGINE_SIG)
    def _op(input_tensors, engine):
        return _execute(input_tensors, engine)

    @_op.register_fake  # type: ignore[misc]
    def _(input_tensors, engine):
        return _execute(input_tensors, engine)

    return _op


vision_tower = _register("vision_tower")
text_encoder = _register("text_encoder")
action_expert = _register("action_expert")
action_context = _register("action_context")

RUNTIME_OPS = {
    "vision_tower": torch.ops.edgellm.vision_tower.default,
    "text_encoder": torch.ops.edgellm.text_encoder.default,
    "action_expert": torch.ops.edgellm.action_expert.default,
    "action_context": torch.ops.edgellm.action_context.default,
}


def specialize_runtime_ops(
    gm: torch.fx.GraphModule, names: Sequence[str]
) -> torch.fx.GraphModule:
    """Rewrite ``execute_engine`` nodes in order onto named ``edgellm::*`` ops."""
    engines = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and "execute_engine" in str(n.target)
    ]
    if len(engines) != len(names):
        raise RuntimeError(
            f"graph has {len(engines)} execute_engine node(s), expected {len(names)} ({list(names)})"
        )
    for node, name in zip(engines, names):
        if name not in RUNTIME_OPS:
            raise KeyError(f"unknown runtime op {name!r}; expected {tuple(RUNTIME_OPS)}")
        node.target = RUNTIME_OPS[name]
    gm.recompile()
    return gm
