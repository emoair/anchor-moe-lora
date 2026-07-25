#!/usr/bin/env python3
"""One-click entry point for the aggregate-only chat five-expert evaluator."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research.gemma3_chat_five_expert_generation_eval_v1 import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
