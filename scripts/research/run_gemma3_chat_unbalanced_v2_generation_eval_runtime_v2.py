"""Repository-local CLI for the unbalanced-v2 physical generation evaluator."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research.gemma3_chat_unbalanced_v2_generation_eval_runtime_v2 import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
