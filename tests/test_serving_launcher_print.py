import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


pytestmark = pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None or shutil.which("wsl.exe") is None,
    reason="PowerShell-to-WSL launcher integration requires the configured Windows host",
)


def _print_command(quantization: str, load_format: str) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / "serve" / "start_vllm.ps1"
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-File",
            str(script),
            "-BaseModel",
            "fake-model",
            "-PlannerAdapter",
            str(ROOT),
            "-ToolPolicyAdapter",
            str(ROOT),
            "-FrontendAdapter",
            str(ROOT),
            "-ReviewAdapter",
            str(ROOT),
            "-SecurityAdapter",
            str(ROOT),
            "-MixedAdapter",
            str(ROOT),
            "-Quantization",
            quantization,
            "-LoadFormat",
            load_format,
            "-PrintCommand",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )


@pytest.mark.parametrize(
    ("quantization", "load_format", "expected"),
    [
        ("bitsandbytes", "bitsandbytes", "--quantization bitsandbytes --load-format bitsandbytes"),
        (
            "compressed-tensors",
            "auto",
            "--quantization compressed-tensors --load-format auto",
        ),
    ],
)
def test_print_command_renders_valid_quantization_pairs(quantization, load_format, expected):
    result = _print_command(quantization, load_format)

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert "qlora" not in result.stdout.lower()


def test_powershell_rejects_crossed_quantization_load_format():
    result = _print_command("bitsandbytes", "auto")

    assert result.returncode != 0
    assert "requires LoadFormat=bitsandbytes" in result.stderr
