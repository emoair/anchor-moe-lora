"""Content-free client for the sealed SWE-bench validator supervisor."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.tooling.swebench_execution_v3 import validator_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(validator_cli())
