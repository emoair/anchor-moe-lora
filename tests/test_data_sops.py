from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.schema import DataValidationError  # noqa: E402
from anchor_mvp.data.sops import load_sop, load_sop_directory  # noqa: E402


def test_loads_markdown_and_yaml_sops() -> None:
    loaded = load_sop_directory(ROOT / "skills")
    assert set(loaded) == {"plan", "tool_policy", "frontend", "review", "security"}
    assert loaded["plan"].sop_id == "implementation-planner-v1"
    assert loaded["tool_policy"].sop_id == "tool-policy-advisory-v1"
    assert loaded["frontend"].sop_id == "frontend-engineering-v1"
    assert loaded["security"].sha256


def test_rejects_unknown_sop_extension(tmp_path: Path) -> None:
    path = tmp_path / "security.txt"
    path.write_text("test", encoding="utf-8")
    with pytest.raises(DataValidationError, match="SOP files"):
        load_sop(path, "security")
