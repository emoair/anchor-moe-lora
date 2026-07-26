"""Repository-local model-free multi-arm entry point."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.training.gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
