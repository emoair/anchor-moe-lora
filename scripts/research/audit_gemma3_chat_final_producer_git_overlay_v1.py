#!/usr/bin/env python3
"""Audit the Gemma 3 chat FINAL Producer Git provenance overlay."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research.gemma3_chat_final_producer_git_overlay_v1 import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
