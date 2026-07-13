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


def test_models_command_reports_missing_key_without_request(
    capsys, monkeypatch
) -> None:
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    assert main(["models", "--provider", "kimi-code-openai"]) == 0
    report = json.loads(capsys.readouterr().out)[0]
    assert report["status"] == "missing_credential"
    assert report["choices"] == []
    assert report["base_url"] == "https://api.kimi.com/coding/v1"


def test_force_model_builds_custom_provider_without_discovery() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--provider",
            "custom-openai",
            "--base-url",
            "https://gateway.example.com/v1",
            "--api-key-env",
            "PRIVATE_TEACHER_KEY",
            "--model",
            "manual-model",
            "--force-model",
        ]
    )
    teacher = _teacher(args, {})
    assert isinstance(teacher, CompatibleTeacher)
    assert teacher.api_key_env == "PRIVATE_TEACHER_KEY"
    assert teacher.provider_provenance == {
        "preset": "custom-openai",
        "base_url": "https://gateway.example.com/v1",
        "protocol": "openai",
        "model": "manual-model",
        "model_source": "manual",
        "discovery": {"status": "skipped_force_model", "model_count": 0},
    }


def test_cli_builds_openai_responses_transport_for_ark_base() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--provider",
            "custom-openai-responses",
            "--base-url",
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "--api-key-env",
            "ARK_TEST_KEY",
            "--model",
            "ark-model-id",
            "--force-model",
            "--protocol",
            "openai_responses",
        ]
    )
    teacher = _teacher(args, {})
    assert isinstance(teacher, CompatibleTeacher)
    assert teacher.protocol == "openai_responses"
    assert teacher.base_url == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert teacher.api_key_env == "ARK_TEST_KEY"
