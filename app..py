from __future__ import annotations
from pathlib import Path

import sys
import argparse
import torch

_TEST_ROOT = Path(__file__).resolve().parents[1]
_VLA_ROOT = Path(__file__).resolve().parent

while str(_VLA_ROOT) in sys.path:
    sys.path.remove(_VLA_ROOT)
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from vla.edge_compile_runner import EdgeCompileRunner
from vla.profiles import MODEL_REGISTRY, get_profile

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True)
    entry, rest = p.parse_known_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    profile = get_profile(entry.model)
    args = EdgeCompileRunner.build_arg_parser(profile).parse_args(rest)
    return EdgeCompileRunner(device, profile, args).run()

if __name__ == "__main__":
    raise SystemExit(main())