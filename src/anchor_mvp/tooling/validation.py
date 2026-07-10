from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .models import ToolTraceEntry, ValidationResult
from .policy import ToolPolicy
from .trace import digest_text


_VALIDATION_NAMES = ("build", "test", "lint")


def _package_scripts(workspace: Path) -> dict[str, str]:
    package_path = workspace / "package.json"
    if not package_path.is_file():
        return {}
    try:
        payload = json.loads(package_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid package.json: {exc}") from exc
    scripts = payload.get("scripts", {})
    if not isinstance(scripts, dict):
        raise ValueError("package.json scripts must be an object")
    return {str(key): str(value) for key, value in scripts.items()}


def run_validations(
    workspace: Path, policy: ToolPolicy
) -> tuple[tuple[ValidationResult, ...], tuple[ToolTraceEntry, ...]]:
    scripts = _package_scripts(workspace)
    results: list[ValidationResult] = []
    trace: list[ToolTraceEntry] = []
    npm_executable = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    for name in _VALIDATION_NAMES:
        command = f"npm run {name} --if-present"
        if not policy.is_command_allowed(command):
            raise ValueError(f"validator command is not whitelisted: {command}")
        if name not in scripts:
            results.append(
                ValidationResult(
                    name=name,
                    command=command,
                    script_present=False,
                    status="SKIP",
                )
            )
            continue
        started = time.perf_counter()
        timed_out = False
        try:
            if npm_executable is None:
                raise FileNotFoundError("npm executable not found")
            completed = subprocess.run(
                [npm_executable, "run", name, "--if-present"],
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=policy.validation_timeout_seconds,
                shell=False,
                check=False,
            )
            exit_code = completed.returncode
            output = completed.stdout
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        except FileNotFoundError:
            exit_code = 127
            output = "npm executable not found"
        duration_ms = (time.perf_counter() - started) * 1000
        output_hash = digest_text(output)
        status = "TIMEOUT" if timed_out else "PASS" if exit_code == 0 else "FAIL"
        results.append(
            ValidationResult(
                name=name,
                command=command,
                script_present=True,
                status=status,
                exit_code=exit_code,
                duration_ms=duration_ms,
                output_sha256=output_hash,
            )
        )
        trace.append(
            ToolTraceEntry(
                sequence=len(trace) + 1,
                source="validator",
                tool="bash",
                status=status.lower(),
                command=command,
                exit_code=exit_code,
                duration_ms=duration_ms,
                output_sha256=output_hash,
            )
        )
    return tuple(results), tuple(trace)
