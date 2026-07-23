#!/usr/bin/env python3
"""Audit the model-free Gemma 3 tokenizer/label binding receipt."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.training.gemma3_tokenizer_binding_v1 import audit_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(audit_main())
