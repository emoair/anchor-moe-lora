"""Small dispatcher for the Anchor-MoE-LoRA subsystems."""

from __future__ import annotations

import runpy
import sys


MODULES = {
    "data": "anchor_mvp.data",
    "train": "anchor_mvp.training",
    "benchmark": "anchor_mvp.benchmark",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        print("usage: python -m anchor_mvp {data|train|benchmark} [arguments...]")
        return 2
    command = sys.argv.pop(1)
    runpy.run_module(MODULES[command], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
