"""Repository-local entry point for the chat five-expert scaffold."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.training.gemma3_chat_five_expert_qonly_v1 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
