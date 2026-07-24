#!/usr/bin/env python3
"""Build the additive Gemma 3 five-expert chat diagnostic dataset."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research.gemma3_chat_five_expert_qonly_v1 import (  # noqa: E402
    build_main,
)


if __name__ == "__main__":
    raise SystemExit(build_main())
