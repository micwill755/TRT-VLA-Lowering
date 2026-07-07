from __future__ import annotations

import argparse
import sys
import logging
import torch_tensorrt

from pathlib import Path

torch_tensorrt.logging.set_level(logging.WARNING)

_TEST_ROOT = Path(__file__).resolve().parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from trt.orchestrator.edge_orchestrator import EdgeOrchestrator
from trt.profile.registry import PROFILE_REGISTRY, get_profile

def build_entry_parser() -> argparse.ArgumentParser:
    known = ", ".join(sorted(PROFILE_REGISTRY))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True, help=f"VLA profile ({known})")
    return parser

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    entry, rest = build_entry_parser().parse_known_args(argv)
    profile = get_profile(entry.model)
    args = EdgeOrchestrator.build_arg_parser(profile).parse_args(rest)
    return EdgeOrchestrator(profile, args).run()

if __name__ == "__main__":
    raise SystemExit(main())
