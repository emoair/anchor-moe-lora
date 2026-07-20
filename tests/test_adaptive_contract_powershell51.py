from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train/run_formal_partial_v1_lowmem.ps1"


@pytest.mark.parametrize(
    ("group", "adapter", "rank"),
    (("E", "planner", 8), ("F", "security_gate", 1)),
)
def test_windows_powershell_51_executes_adaptive_contract_without_training(
    group: str,
    adapter: str,
    rank: int,
) -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    temporary = Path(tempfile.gettempdir())
    before = set(temporary.glob("anchor-adaptive-contract-*.py"))

    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-Arm",
            "train",
            "-Group",
            group,
            "-Adapter",
            adapter,
            "-AllowLowHostMemory",
            "-AdaptiveContractOnly",
            "-Python",
            sys.executable,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["Mode"] == "adaptive-contract-only"
    assert value["Group"] == group
    assert value["Adapter"] == adapter
    assert value["Rank"] == rank
    assert set(temporary.glob("anchor-adaptive-contract-*.py")) == before
    assert "preflight" not in result.stdout.lower()
