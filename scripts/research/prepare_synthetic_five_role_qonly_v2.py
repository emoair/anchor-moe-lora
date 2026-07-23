from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anchor_mvp.training.qwen_synthetic_five_role_qonly_v2 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
