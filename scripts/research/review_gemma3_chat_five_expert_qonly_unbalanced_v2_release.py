"""Run the independent sharded Gemma 3 Chat v2 release review."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research import (  # noqa: E402
    gemma3_chat_five_expert_qonly_unbalanced_v2_release as release,
)


if __name__ == "__main__":
    raise SystemExit(release.main())
