from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/data/start_automation.ps1"
ANCHOR = ROOT / "anchor.ps1"
FORMAL_GATE_READER = ROOT / "scripts/observability/formal_gate_status.py"


def _powershell() -> str:
    executable = shutil.which("powershell.exe")
    if executable is None:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    return executable


def test_launcher_has_no_provider_specific_credential_or_python_command() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "$env:KIMI_API_KEY" not in source
    assert "provider_spec(raw).api_key_env" in source
    assert "-PromptForApiKey" in source
    assert "& python " not in source


def test_windows_powershell_51_validates_generic_provider_config(
    tmp_path: Path,
) -> None:
    config = tmp_path / "automation-generic-provider.yaml"
    config.write_text(
        (ROOT / "configs/data/automation.mock.yaml").read_text(encoding="utf-8")
        + "\n"
        + "provider: custom-openai-responses\n"
        + "protocol: openai_responses\n"
        + "base_url: https://example.invalid/api/coding/v3\n"
        + "model: offline-model\n"
        + "force_model: true\n"
        + "discover_models: false\n"
        + "api_key_env: ARK_OFFLINE_TEST_KEY\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("ARK_OFFLINE_TEST_KEY", None)

    result = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-Config",
            str(config),
            "-PythonExe",
            sys.executable,
            "-ValidateConfig",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Credential environment: ARK_OFFLINE_TEST_KEY (value hidden)" in result.stdout
    assert "Configuration validated. No provider request was sent." in result.stdout
    assert "example.invalid" not in result.stdout


def test_windows_powershell_51_rejects_live_run_without_declared_credential(
    tmp_path: Path,
) -> None:
    config = tmp_path / "automation-missing-credential.yaml"
    config.write_text(
        (ROOT / "configs/data/automation.mock.yaml").read_text(encoding="utf-8")
        + "\n"
        + "provider: custom-openai-responses\n"
        + "protocol: openai_responses\n"
        + "base_url: https://example.invalid/api/coding/v3\n"
        + "model: offline-model\n"
        + "force_model: true\n"
        + "discover_models: false\n"
        + "api_key_env: GENERIC_OFFLINE_TEST_KEY\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("GENERIC_OFFLINE_TEST_KEY", None)

    result = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-Config",
            str(config),
            "-PythonExe",
            sys.executable,
            "-NoWaitCooldown",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "GENERIC_OFFLINE_TEST_KEY" in result.stderr
    assert "pass -PromptForApiKey" in result.stderr
    assert "example.invalid" not in result.stderr


def test_anchor_entrypoint_exposes_ui_and_distillation_boundaries() -> None:
    source = ANCHOR.read_text(encoding="utf-8-sig")

    assert '"status"' in source
    assert '"ui"' in source
    assert '"preflight"' in source
    assert '"distill-swebench"' in source
    assert '"distill-synthetic"' in source
    assert "[switch]$ConfirmLive" in source
    assert '"--confirm-live"' in source
    assert "run_swebench_ccswitch.py" in source
    assert "-WindowStyle Hidden" in source
    assert "never falls back to synthetic direct" in source
    assert "Invoke-ContentFreeFormalPreflight" in source
    assert "Official heldout eval gate" in source
    assert "NON-BLOCKING" in source
    assert '"sample_bodies_read"' in source
    assert "current npm-only v2" not in source


def test_formal_gate_reader_is_content_free_and_fail_closed() -> None:
    environment = os.environ.copy()
    environment["ARK_CODING_API_KEY"] = "DO-NOT-PRINT-ARK-CREDENTIAL"
    environment["KIMI_API_KEY"] = "DO-NOT-PRINT-KIMI-CREDENTIAL"
    result = subprocess.run(
        [sys.executable, str(FORMAL_GATE_READER), "--root", str(ROOT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "anchor.swebench-ccswitch-preflight.v1"
    assert payload["content_free"] is True
    assert payload["provider_requests"] == 0
    assert payload["credentials_read"] is False
    assert payload["sample_bodies_read"] is False
    assert payload["sample_bodies_printed"] is False
    assert payload["heldout_files_read"] is False
    assert payload["live_started"] is False
    serialized = json.dumps(payload)
    assert "DO-NOT-PRINT-ARK-CREDENTIAL" not in serialized
    assert "DO-NOT-PRINT-KIMI-CREDENTIAL" not in serialized


def test_anchor_swebench_refuses_any_incomplete_formal_chain_without_fallback() -> None:
    result = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ANCHOR),
            "-Action",
            "distill-swebench",
            "-SWEConfig",
            "configs/data/definitely-missing-for-fail-closed-test.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert result.returncode == 4
    assert "Launch refused" in result.stdout
    assert "CC Switch-bound SWE-bench config" in result.stdout
    assert "never falls back to synthetic direct" in result.stdout


def test_anchor_swebench_defaults_to_zero_request_offline_preflight() -> None:
    environment = os.environ.copy()
    environment.pop("ARK_CODING_API_KEY", None)
    environment.pop("KIMI_API_KEY", None)

    result = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ANCHOR),
            "-Action",
            "distill-swebench",
            "-PythonExe",
            sys.executable,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"provider_requests": 0' in result.stdout
    assert '"credentials_read": false' in result.stdout
    assert '"live_started": false' in result.stdout
    assert "not wired" not in result.stdout
