from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "scripts"
    / "research"
    / "run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1.ps1"
)
PYTHON_ENTRYPOINT = LAUNCHER.with_suffix(".py")
CONFIG = (
    ROOT
    / "configs"
    / "training"
    / "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1.yaml"
)


def test_powershell_launcher_is_model_free_and_has_no_execute_path() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "\r" not in text
    assert "nvidia-smi" not in lowered
    assert "cuda_visible_devices" not in lowered
    assert "formal-v3-training.lock" not in lowered
    assert "start-process" not in lowered
    assert "invoke-webrequest" not in lowered
    assert "invoke-restmethod" not in lowered
    assert "[switch]$execute" not in lowered
    assert "[switch]$dryrun" in lowered
    assert "[string]$consumerpreflightreceipt" in lowered
    assert "[string]$tensorinventory" in lowered
    assert "--consumer-preflight-receipt" in lowered
    assert "--tensor-inventory" in lowered


def test_launcher_paths_are_exact_and_repository_local() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert PYTHON_ENTRYPOINT.name in text
    assert CONFIG.name in text
    assert "run_gemma3_chat_five_expert_qonly_v1" not in text
    assert "gemma3_1b_it_five_role_bf16_lora_v2" not in text
    assert "gemma3_1b_it_five_role_q8_qlora_v2" not in text
    assert PYTHON_ENTRYPOINT.is_file()
    assert CONFIG.is_file()


def test_powershell_validate_config_executes_without_gpu_or_model() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell capability unavailable")
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-ValidateConfig",
            "-Python",
            sys.executable,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status":"passed"' in completed.stdout
    assert '"model_loaded":false' in completed.stdout
    assert '"gpu_touched":false' in completed.stdout
    assert '"network_used":false' in completed.stdout
