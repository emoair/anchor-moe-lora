from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.cli import _teacher, build_parser, main  # noqa: E402
from anchor_mvp.data.teacher import CompatibleTeacher  # noqa: E402


def test_dry_run_cli(capsys, tmp_path: Path) -> None:
    status = main(
        [
            "run",
            "--dry-run",
            "--seed-count",
            "2",
            "--concurrency",
            "2",
            "--sop-dir",
            str(ROOT / "skills"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report["written_by_task"] == {
        "plan": 2,
        "tool_policy": 2,
        "frontend": 2,
        "review": 2,
        "security": 2,
    }


def test_mock_probe_cli(capsys) -> None:
    assert main(["probe", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == [{"ok": True, "model": "mock-teacher-v1", "protocol": "mock"}]


def test_cli_exposes_timeout_and_retry_policy() -> None:
    args = build_parser().parse_args(
        [
            "probe",
            "--timeout-seconds",
            "720.5",
            "--max-retries",
            "0",
            "--stream-openai",
            "--stream-options-include-usage",
        ]
    )
    teacher = _teacher(args, {})
    assert isinstance(teacher, CompatibleTeacher)
    assert teacher.timeout_seconds == 720.5
    assert teacher.max_retries == 0
    assert teacher.stream_openai is True
    assert teacher.stream_options_include_usage is True
