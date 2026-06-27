"""Unified Edge-LLM compile entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from vla.base_compile_edge_llm import BaseEdgeCompileRunner
from vla.profiles import get_profile

def _pop_model_arg(argv: list[str]) -> tuple[str, list[str]]:
    model = "groot"
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--model":
            model = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return model, cleaned

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    model, argv = _pop_model_arg(argv)
    profile = get_profile(model)
    args = BaseEdgeCompileRunner.parse_args_for_profile(profile, argv)
    return BaseEdgeCompileRunner(profile, args).run()

if __name__ == "__main__":
    raise SystemExit(main())