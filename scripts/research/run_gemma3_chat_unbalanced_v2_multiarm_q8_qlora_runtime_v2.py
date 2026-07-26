"""CLI wrapper for the explicit Gemma 3 unbalanced-v2 multi-arm runtime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchor_mvp.training.gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2 import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
