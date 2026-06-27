"""Unified Edge-LLM compile entrypoint.

This script is the single CLI for exporting and benchmarking all supported VLAs
(groot, pi05, smolvla, molmoact2). It selects a profile by ``--model``, then
delegates to ``BaseEdgeCompileRunner`` for load → export → benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure ``Test/`` is on sys.path when invoked as ``python vla/compile_edge_llm.py``.
_TEST_ROOT = Path(__file__).resolve().parents[1]
_VLA_ROOT = Path(__file__).resolve().parent
_vla_root = str(_VLA_ROOT)
# ``python vla/compile_edge_llm.py`` adds ``Test/vla/`` to sys.path, which shadows
# stdlib ``profile`` via ``vla/profile.py`` when torch imports ``cProfile``.
while _vla_root in sys.path:
    sys.path.remove(_vla_root)
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from vla.base_compile_edge_llm import BaseEdgeCompileRunner
from vla.profiles import MODEL_REGISTRY, get_profile

def build_entry_parser() -> argparse.ArgumentParser:
    """Build the first-pass parser that only resolves ``--model``.

    Profile-specific flags (``--model-id``, ``--engine-dir``, etc.) are registered
    later by ``BaseEdgeCompileRunner.build_arg_parser(profile)``, so we parse
    ``--model`` first to know which profile class to use.
    """
    known = ", ".join(sorted(MODEL_REGISTRY))
    parser = argparse.ArgumentParser(
        description="Export VLA TensorRT engines for TensorRT-Edge-LLM",
        # Defer ``-h`` to the profile-specific parser so users see the full flag set.
        add_help=False,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="NAME",
        help=f"VLA profile name ({known})",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, resolve the VLA profile, and run export + benchmark."""
    # Allow tests to inject argv; default to everything after the script name.
    argv = list(sys.argv[1:] if argv is None else argv)

    # Phase 1: require ``--model`` and leave other flags for the profile parser.
    entry_args, remaining = build_entry_parser().parse_known_args(argv)
    profile = get_profile(entry_args.model)

    # Phase 2: parse profile-specific options (model-id, engine-dir, export flags, …).
    args = BaseEdgeCompileRunner.parse_args_for_profile(profile, remaining)
    return BaseEdgeCompileRunner(profile, args).run()

if __name__ == "__main__":
    raise SystemExit(main())