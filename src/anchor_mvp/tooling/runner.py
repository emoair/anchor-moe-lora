from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import threading
from contextlib import nullcontext
from typing import Mapping, Protocol

from .config import AGENT_ID, DEFAULT_MODEL, DEFAULT_VARIANT, PROVIDER_ID
from .behavioral_probe import run_behavioral_probe
from .initial_tool_proxy import InitialToolChoiceProxy
from .models import AgentExecution, ToolTraceEntry
from .opencode_artifact import (
    BinaryAttestation,
    verify_binary_attestation,
    verify_binary_identity,
    verify_launch_identity,
)
from .policy import ToolPolicy
from .trace import (
    classify_error_metadata,
    digest_text,
    parse_opencode_jsonl,
    parse_public_outcome,
)
from .session_export import (
    QuarantineError,
    SessionConversionPolicy,
    append_jsonl,
    convert_controlled_session,
    quarantine_record,
)


_CAPTURE_LOCK = threading.Lock()
_SESSION_ID = __import__("re").compile(r"^ses_[A-Za-z0-9_-]{4,128}$")


@dataclass(frozen=True)
class ControlledSessionCapture:
    candidates_path: Path
    quarantine_path: Path
    heldout_cases: Path
    heldout_fixtures_root: Path
    heldout_manifest: Path

    def __post_init__(self) -> None:
        for value in (
            self.candidates_path,
            self.quarantine_path,
            self.heldout_cases,
            self.heldout_fixtures_root,
            self.heldout_manifest,
        ):
            if not value.is_absolute():
                raise ValueError("controlled session capture paths must be absolute")


def _walk_objects(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _extract_session_id(stdout: str) -> str | None:
    found: set[str] = set()
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        for item in _walk_objects(value):
            for name in ("sessionID", "session_id", "sessionId"):
                candidate = item.get(name)
                if isinstance(candidate, str) and _SESSION_ID.fullmatch(candidate):
                    found.add(candidate)
    return next(iter(found)) if len(found) == 1 else None


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
    def __init__(
        self,
        executable: str = "opencode",
        *,
        extra_environment: Mapping[str, str] | None = None,
        initial_tool_proxy: bool = False,
        patch_manifest: str | Path | None = None,
        session_capture: ControlledSessionCapture | None = None,
    ) -> None:
        self.executable = executable
        self.extra_environment = dict(extra_environment or {})
        self.initial_tool_proxy = initial_tool_proxy
        project_root = Path(__file__).resolve().parents[3]
        self.patch_manifest = Path(
            patch_manifest or project_root / "patches" / "opencode" / "patch-manifest.json"
        ).resolve()
        self.session_capture = session_capture
        self._attestation: BinaryAttestation | None = None

    @property
    def backend_name(self) -> str:
        return (
            "opencode-kimi-initial-tool-proxy"
            if self.initial_tool_proxy
            else "opencode-kimi"
        )

    def _resolved_executable(self) -> str | None:
        if os.name == "nt" and not Path(self.executable).suffix:
            command_shim = shutil.which(f"{self.executable}.cmd")
            if command_shim:
                return command_shim
        return shutil.which(self.executable)

    def available(self) -> bool:
        return self._resolved_executable() is not None

    @staticmethod
    def is_patched_agent_config(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        options = value.get("options")
        return value.get("requireInitialToolCall") is True and not (
            isinstance(options, dict) and "requireInitialToolCall" in options
        )

    def probe_patched(self, config_path: Path) -> tuple[bool, str]:
        executable = self._resolved_executable()
        if executable is None:
            return False, "patched executable is missing"
        try:
            attestation = verify_binary_attestation(
                executable, patch_manifest=self.patch_manifest
            )
        except (OSError, ValueError) as error:
            return False, str(error)
        environment = self._environment(config_path)
        environment.pop("KIMI_CODE_API_KEY", None)
        try:
            verify_binary_identity(attestation)
            completed = subprocess.run(
                [str(attestation.executable), "debug", "agent", AGENT_ID, "--pure"],
                cwd=config_path.parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                return False, f"patched capability probe exited {completed.returncode}"
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                return False, "patched capability probe returned invalid JSON"
            if not self.is_patched_agent_config(payload):
                return False, "requireInitialToolCall is not a first-class resolved agent field"
            verify_binary_identity(attestation)
            with tempfile.TemporaryDirectory(
                prefix="opencode-behavioral-probe-",
                dir=config_path.parent.parent,
            ) as probe_root:
                passed, reason = run_behavioral_probe(
                    attestation.executable,
                    probe_root=Path(probe_root),
                    environment=environment,
                )
            if not passed:
                return False, reason
            self._attestation = attestation.with_behavioral_probe()
            return True, reason
        except subprocess.TimeoutExpired:
            return False, "patched capability probe timed out"
        finally:
            shutil.rmtree(config_path.parent / "runtime", ignore_errors=True)

    def probe_initial_tool_proxy(self, config_path: Path) -> tuple[bool, str]:
        if not self.initial_tool_proxy:
            return False, "initial-tool proxy mode is disabled"
        if self._resolved_executable() is None:
            return False, "OpenCode executable is missing"
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            provider = loaded["provider"][PROVIDER_ID]
            if not isinstance(provider, dict) or not isinstance(
                provider.get("options"), dict
            ):
                return False, "OpenCode provider configuration is invalid"
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return False, "OpenCode provider configuration is invalid"
        return True, "loopback initial-tool proxy is configured"

    @staticmethod
    def _proxy_config(config_path: Path, base_url: str) -> Path:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        provider = loaded["provider"][PROVIDER_ID]
        provider["options"]["baseURL"] = base_url
        agent = loaded.get("agent", {}).get(AGENT_ID)
        if isinstance(agent, dict):
            # The unpatched binary would treat unknown agent keys as provider
            # options. Enforcement belongs exclusively to the proxy fallback.
            agent.pop("requireInitialToolCall", None)
        destination = config_path.with_name("opencode.proxy.json")
        destination.write_text(
            json.dumps(loaded, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    def _environment(self, config_path: Path) -> dict[str, str]:
        runtime_root = config_path.parent / "runtime"
        config_root = runtime_root / "config"
        data_root = runtime_root / "data"
        cache_root = runtime_root / "cache"
        for directory in (config_root, data_root, cache_root):
            directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(self.extra_environment)
        environment.pop("OPENCODE_CONFIG_CONTENT", None)
        environment.update(
            {
                "OPENCODE_CONFIG": str(config_path),
                "OPENCODE_CONFIG_DIR": str(config_root),
                "OPENCODE_CLIENT": "cli",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
                "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
                "OPENCODE_DISABLE_CLAUDE_CODE": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
                "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                "OPENCODE_AUTO_SHARE": "false",
                "OPENCODE_ENABLE_EXA": "false",
                "XDG_CONFIG_HOME": str(config_root),
                "XDG_DATA_HOME": str(data_root),
                "XDG_CACHE_HOME": str(cache_root),
            }
        )
        return environment

    def _verified_executable(self) -> Path:
        if self._attestation is None:
            raise ValueError("patched OpenCode behavioral attestation is missing")
        verify_launch_identity(self._attestation)
        return self._attestation.executable

    def command(self, *, sample_id: str, prompt: str, workspace: Path) -> list[str]:
        executable = self._verified_executable()
        return [
            str(executable),
            "run",
            "--format",
            "json",
            "--model",
            f"{PROVIDER_ID}/{DEFAULT_MODEL}",
            "--agent",
            AGENT_ID,
            "--variant",
            DEFAULT_VARIANT,
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
        try:
            self._verified_executable()
        except (OSError, ValueError):
            return AgentExecution(
                exit_code=126,
                timed_out=False,
                duration_ms=0.0,
                error_codes=("binary_attestation_missing_or_changed",),
            )
        started = time.perf_counter()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proxy_context = InitialToolChoiceProxy() if self.initial_tool_proxy else nullcontext()
        proxy_not_forced = False
        proxy_error_codes: tuple[str, ...] = ()
        with proxy_context as proxy:
            execution_config = config_path
            if isinstance(proxy, InitialToolChoiceProxy):
                execution_config = self._proxy_config(config_path, proxy.base_url)
            verify_launch_identity(self._attestation)  # type: ignore[arg-type]
            process = subprocess.Popen(
                self.command(sample_id=sample_id, prompt=prompt, workspace=workspace),
                cwd=workspace,
                env=self._environment(execution_config),
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
            if isinstance(proxy, InitialToolChoiceProxy):
                stats = proxy.stats
                proxy_not_forced = stats.requests > 0 and stats.forced_requests == 0
                proxy_error_codes = stats.error_codes
        duration_ms = (time.perf_counter() - started) * 1000
        trace, rejected = parse_opencode_jsonl(stdout, policy)
        public_outcome = parse_public_outcome(stdout)
        errors = list(classify_error_metadata(stdout, stderr))
        errors.extend(proxy_error_codes)
        runtime_path = config_path.parent / "runtime"
        session_id = _extract_session_id(stdout)
        export_path: Path | None = None
        if session_id is not None:
            try:
                executable = self._verified_executable()
                exported = subprocess.run(
                    [str(executable), "export", session_id],
                    cwd=workspace,
                    env=self._environment(execution_config),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    timeout=30,
                    check=False,
                )
                if exported.returncode == 0:
                    export_path = runtime_path / "capture" / "session.raw.json"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    export_path.write_bytes(exported.stdout)
                else:
                    errors.append("controlled_session_export_failed")
            except (OSError, subprocess.TimeoutExpired, ValueError):
                errors.append("controlled_session_export_failed")
        else:
            errors.append("controlled_session_id_missing")
        if timed_out:
            errors.append("wrapper_timeout")
        if proxy_not_forced:
            errors.append("initial_tool_proxy_not_forced")
        return AgentExecution(
            exit_code=process.returncode if process.returncode is not None else 124,
            timed_out=timed_out,
            duration_ms=duration_ms,
            trace=trace,
            stdout_sha256=digest_text(stdout),
            stderr_sha256=digest_text(stderr),
            rejected_events=rejected,
            error_codes=tuple(dict.fromkeys(errors)),
            public_outcome=public_outcome,
            controlled_session_id=session_id,
            controlled_export_path=str(export_path) if export_path else None,
            isolated_runtime_path=str(runtime_path),
            opencode_version=(
                self._attestation.opencode_version if self._attestation is not None else None
            ),
        )

    def finalize_capture(
        self,
        *,
        execution: AgentExecution,
        sample_id: str,
        workspace: Path,
        validators: tuple[dict[str, object], ...],
    ) -> tuple[bool, str | None]:
        runtime = Path(execution.isolated_runtime_path) if execution.isolated_runtime_path else None
        export_path = Path(execution.controlled_export_path) if execution.controlled_export_path else None
        export_bytes = export_path.read_bytes() if export_path and export_path.is_file() else b""
        try:
            if self.session_capture is None:
                raise QuarantineError("session_capture_not_configured")
            if not export_bytes or execution.controlled_session_id is None:
                raise QuarantineError("controlled_export_missing")
            try:
                export_data = json.loads(export_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise QuarantineError("invalid_json_or_encoding") from error
            if not isinstance(export_data, dict):
                raise QuarantineError("input_not_object")
            outcome = execution.public_outcome
            capture = {
                "schema_version": "anchor.controlled-session-capture.v1",
                "source": "opencode-export-controlled-fixture",
                "sample_id": sample_id,
                "session_id": execution.controlled_session_id,
                "opencode_version": execution.opencode_version,
                "validators": list(validators),
                "public_outcome": asdict(outcome) if outcome is not None else None,
            }
            candidate = convert_controlled_session(
                export_data,
                capture,
                SessionConversionPolicy(
                    workspace_root=workspace.resolve(),
                    heldout_cases=self.session_capture.heldout_cases,
                    heldout_fixtures_root=self.session_capture.heldout_fixtures_root,
                    heldout_manifest=self.session_capture.heldout_manifest,
                ),
            )
            with _CAPTURE_LOCK:
                append_jsonl(self.session_capture.candidates_path, candidate)
            return True, None
        except (OSError, ValueError) as error:
            code = error.code if isinstance(error, QuarantineError) else "conversion_error"
            if self.session_capture is not None:
                with _CAPTURE_LOCK:
                    append_jsonl(
                        self.session_capture.quarantine_path,
                        quarantine_record(
                            sample_id=sample_id,
                            code=code,
                            export_bytes=export_bytes,
                        ),
                    )
            return False, code
        finally:
            if runtime is not None:
                shutil.rmtree(runtime, ignore_errors=True)


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
        public_outcome=None,
    ) -> None:
        self.file_updates = dict(file_updates or {})
        self.commands = commands
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.public_outcome = public_outcome

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
            public_outcome=self.public_outcome,
        )
