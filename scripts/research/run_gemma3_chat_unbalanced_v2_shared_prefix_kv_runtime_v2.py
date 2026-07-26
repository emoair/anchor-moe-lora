"""CLI shim for the Gemma 3 shared-prefix KV runtime v2."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research.gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2 import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
