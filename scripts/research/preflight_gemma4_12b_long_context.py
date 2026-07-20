"""Thin entry point for the static Gemma 4 12B long-context preflight."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anchor_mvp.research.long_context_preflight import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
