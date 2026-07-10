from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Mapping, Protocol

from .config import AGENT_ID, DEFAULT_MODEL, PROVIDER_ID
from .models import AgentExecution, ToolTraceEntry
from .policy import ToolPolicy
from .trace import classify_error_text, digest_text, parse_opencode_jsonl


class AgentExecutor(Protocol):
    backend_name: str

    def run(
        self,
        *,
        sample_id: str,
        prompt: str,
        workspace: Path,
        config_path: Path,
        policy: ToolPolicy,
    ) -> AgentExecution:
        ...


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class OpenCodeExecutor:
    backend_name = "opencode-kimi"

    def __init__(
        self,
        executable: str = "opencode",
        *,
        extra_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = executable
        self.extra_environment = dict(extra_environment or {})

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _environment(self, config_path: Path) -> dict[str, str]:
        runtime_root = config_path.parent / "runtime"
        config_root = runtime_root / "config"
        data_root = runtime_root / "data"
        cache_root = runtime_root / "cache"
        for directory in (config_root, data_root, cache_root):
            directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(self.extra_environment)
        environment.update(
            {
                "OPENCODE_CONFIG": str(config_path),
                "OPENCODE_CONFIG_DIR": str(config_root),
                "OPENCODE_CLIENT": "cli",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
                "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
                "OPENCODE_DISABLE_CLAUDE_CODE": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
                "OPENCODE_AUTO_SHARE": "false",
                "OPENCODE_ENABLE_EXA": "false",
                "XDG_CONFIG_HOME": str(config_root),
                "XDG_DATA_HOME": str(data_root),
                "XDG_CACHE_HOME": str(cache_root),
            }
        )
        return environment

    def command(self, *, sample_id: str, prompt: str, workspace: Path) -> list[str]:
        return [
            self.executable,
            "run",
            "--format",
            "json",
            "--model",
            f"{PROVIDER_ID}/{DEFAULT_MODEL}",
            "--agent",
            AGENT_ID,
            "--dir",
            str(workspace),
            "--title",
            f"anchor-gold:{sample_id}",
            prompt,
        ]

    def run(
        self,
        *,
        sample_id: str,
        prompt: str,
        workspace: Path,
        config_path: Path,
        policy: ToolPolicy,
    ) -> AgentExecution:
        if not self.available():
            return AgentExecution(
                exit_code=127,
                timed_out=False,
                duration_ms=0.0,
                error_codes=("opencode_not_installed",),
            )
        started = time.perf_counter()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            self.command(sample_id=sample_id, prompt=prompt, workspace=workspace),
            cwd=workspace,
            env=self._environment(config_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=policy.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
            stdout, stderr = process.communicate()
        duration_ms = (time.perf_counter() - started) * 1000
        trace, rejected = parse_opencode_jsonl(stdout, policy)
        errors = list(classify_error_text(stdout + "\n" + stderr))
        # OpenCode persists sessions by default. Its XDG roots are redirected
        # above and removed after event reduction so hidden reasoning/session
        # bodies cannot become training artifacts.
        shutil.rmtree(config_path.parent / "runtime", ignore_errors=True)
        if timed_out:
            errors.append("wrapper_timeout")
        return AgentExecution(
            exit_code=process.returncode if process.returncode is not None else 124,
            timed_out=timed_out,
            duration_ms=duration_ms,
            trace=trace,
            stdout_sha256=digest_text(stdout),
            stderr_sha256=digest_text(stderr),
            rejected_events=rejected,
            error_codes=tuple(dict.fromkeys(errors)),
        )


class MockAgentExecutor:
    """Deterministic executor for unit tests and offline pipeline smoke tests."""

    backend_name = "mock-opencode"

    def __init__(
        self,
        *,
        file_updates: Mapping[str, str] | None = None,
        commands: tuple[str, ...] = (),
        exit_code: int = 0,
        timed_out: bool = False,
    ) -> None:
        self.file_updates = dict(file_updates or {})
        self.commands = commands
        self.exit_code = exit_code
        self.timed_out = timed_out

    def run(
        self,
        *,
        sample_id: str,
        prompt: str,
        workspace: Path,
        config_path: Path,
        policy: ToolPolicy,
    ) -> AgentExecution:
        started = time.perf_counter()
        trace: list[ToolTraceEntry] = []
        rejected = 0
        root = workspace.resolve()
        for relative, content in self.file_updates.items():
            destination = (root / relative).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"mock update escapes workspace: {relative}") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            trace.append(
                ToolTraceEntry(
                    sequence=len(trace) + 1,
                    source="agent",
                    tool="edit",
                    status="completed",
                )
            )
        for command in self.commands:
            allowed = policy.is_command_allowed(command)
            if not allowed:
                rejected += 1
            trace.append(
                ToolTraceEntry(
                    sequence=len(trace) + 1,
                    source="agent",
                    tool="bash",
                    status="completed" if allowed else "rejected",
                    command=policy.normalize_command(command) if allowed else None,
                    command_sha256=None if allowed else policy.command_digest(command),
                    exit_code=0 if allowed else None,
                )
            )
        duration_ms = (time.perf_counter() - started) * 1000
        errors = ("wrapper_timeout",) if self.timed_out else ()
        return AgentExecution(
            exit_code=self.exit_code,
            timed_out=self.timed_out,
            duration_ms=duration_ms,
            trace=tuple(trace),
            rejected_events=rejected,
            error_codes=errors,
        )
