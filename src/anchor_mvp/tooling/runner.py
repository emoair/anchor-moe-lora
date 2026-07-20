from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import threading
from typing import Mapping, Protocol

from .config import (
    AGENT_ID,
    DEFAULT_PROVIDER,
    OpenCodeProvider,
    SANDBOX_API_KEY_ENV,
)
from .behavioral_probe import run_behavioral_probe
from .models import AgentExecution, PublicOutcome, SkillProvenance, ToolTraceEntry
from .opencode_artifact import (
    BinaryAttestation,
    verify_binary_attestation,
    verify_binary_identity,
    verify_launch_identity,
)
from .policy import ToolPolicy
from .responses_wire_probe import run_responses_wire_probe
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
    convert_controlled_session_staging,
    credential_fingerprint,
    quarantine_record,
)
from .workspace import safe_sample_id


_CAPTURE_LOCK = threading.Lock()
_SESSION_ID = __import__("re").compile(r"^ses_[A-Za-z0-9_-]{4,128}$")
_MEMORY_LIMIT = re.compile(r"^[1-9][0-9]*(?:[KMGTP](?:i?B)?|[kmg])$")
_CPU_LIMIT = re.compile(r"^(?:0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(?:\.[0-9]+)?)$")
_AUDITED_ROUTE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
MIN_PROBE_TIMEOUT_SECONDS = 150.0
DEFAULT_PROBE_TIMEOUT_SECONDS = 300.0


def _public_outcome_capture(outcome: PublicOutcome | None) -> dict[str, object] | None:
    """Encode the trusted sidecar as JSON-native values without widening the gate."""

    if outcome is None:
        return None
    return {
        "schema_version": outcome.schema_version,
        "status": outcome.status,
        "decision_trace": [
            {
                "check": step.check,
                "evidence": step.evidence,
                "action": step.action,
            }
            for step in outcome.decision_trace
        ],
        "repair_summaries": list(outcome.repair_summaries),
        "final_summary": outcome.final_summary,
    }


@dataclass(frozen=True)
class ControlledSessionCapture:
    candidates_path: Path
    quarantine_path: Path
    heldout_cases: Path
    heldout_fixtures_root: Path
    heldout_manifest: Path
    staging_path: Path | None = None
    mode: str = "strict"

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
        if self.staging_path is not None and not self.staging_path.is_absolute():
            raise ValueError("controlled session staging path must be absolute")
        if self.mode not in {"strict", "collect"}:
            raise ValueError(
                "controlled session capture mode must be strict or collect"
            )
        if self.mode == "collect" and self.staging_path is None:
            raise ValueError("collect mode requires a staging path")


@dataclass(frozen=True)
class AnchorSandboxOptions:
    """Operator-owned limits for the patched OpenCode anchor sandbox command."""

    linux_executable: Path | None = None
    wsl_distro: str | None = None
    supervisor: str | None = None
    memory: str | None = None
    cpus: str | None = None
    pids: int | None = None
    timeout_seconds: int | None = None
    route_host: str | None = None
    route_port: int | None = None

    def __post_init__(self) -> None:
        if (
            self.linux_executable is not None
            and not self.linux_executable.is_absolute()
        ):
            raise ValueError("anchor sandbox linux_executable must be absolute")
        if self.wsl_distro is not None and not self.wsl_distro.strip():
            raise ValueError("anchor sandbox wsl_distro must not be blank")
        if self.supervisor not in {None, "direct", "wsl-root-systemd"}:
            raise ValueError(
                "anchor sandbox supervisor must be direct or wsl-root-systemd"
            )
        if self.memory is not None and not _MEMORY_LIMIT.fullmatch(self.memory):
            raise ValueError("anchor sandbox memory must be a positive size such as 4G")
        if self.cpus is not None and not _CPU_LIMIT.fullmatch(self.cpus):
            raise ValueError("anchor sandbox cpus must be a positive decimal")
        for value, name in (
            (self.pids, "pids"),
            (self.timeout_seconds, "timeout_seconds"),
            (self.route_port, "route_port"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"anchor sandbox {name} must be a positive integer")
        if self.route_port is not None and self.route_port > 65535:
            raise ValueError("anchor sandbox route_port must be at most 65535")
        if self.route_host is not None:
            try:
                address = ipaddress.ip_address(self.route_host)
            except ValueError as exc:
                raise ValueError(
                    "anchor sandbox route_host must be a literal local IP"
                ) from exc
            if not isinstance(address, ipaddress.IPv4Address) or not any(
                address in network for network in _AUDITED_ROUTE_NETWORKS
            ):
                raise ValueError("anchor sandbox route_host must be local")
        if (self.route_host is None) != (self.route_port is None):
            raise ValueError(
                "anchor sandbox route_host and route_port must be supplied together"
            )

    def command_options(self) -> list[str]:
        result: list[str] = []
        if self.linux_executable is not None:
            result.extend(("--linux-executable", str(self.linux_executable)))
        if self.wsl_distro is not None:
            result.extend(("--wsl-distro", self.wsl_distro))
        if self.supervisor is not None:
            result.extend(("--supervisor", self.supervisor))
        if self.memory is not None:
            result.extend(("--memory", self.memory))
        if self.cpus is not None:
            result.extend(("--cpus", self.cpus))
        if self.pids is not None:
            result.extend(("--pids", str(self.pids)))
        if self.timeout_seconds is not None:
            result.extend(("--timeout", str(self.timeout_seconds)))
        if self.route_host is not None and self.route_port is not None:
            result.extend(("--route-host", self.route_host))
            result.extend(("--route-port", str(self.route_port)))
        return result

    def cleanup_command_options(self) -> list[str]:
        """Return only the routing flags accepted by ``opencode anchor cleanup``."""

        result: list[str] = []
        if self.wsl_distro is not None:
            result.extend(("--wsl-distro", self.wsl_distro))
        if self.supervisor is not None:
            result.extend(("--supervisor", self.supervisor))
        return result


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
    @property
    def backend_name(self) -> str: ...

    def run(
        self,
        *,
        sample_id: str,
        prompt: str,
        workspace: Path,
        config_path: Path,
        policy: ToolPolicy,
    ) -> AgentExecution: ...


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
        killpg = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", None)
        if callable(killpg) and sigkill is not None:
            try:
                killpg(process.pid, sigkill)
            except ProcessLookupError:
                pass


class OpenCodeExecutor:
    def __init__(
        self,
        executable: str = "opencode",
        *,
        extra_environment: Mapping[str, str] | None = None,
        patch_manifest: str | Path | None = None,
        session_capture: ControlledSessionCapture | None = None,
        sandbox_options: AnchorSandboxOptions | None = None,
        provider: OpenCodeProvider = DEFAULT_PROVIDER,
        probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(provider, OpenCodeProvider):
            raise ValueError("provider must be an audited OpenCodeProvider")
        if (
            isinstance(probe_timeout_seconds, bool)
            or not isinstance(probe_timeout_seconds, (int, float))
            or probe_timeout_seconds < MIN_PROBE_TIMEOUT_SECONDS
        ):
            raise ValueError("probe_timeout_seconds must be at least 150 seconds")
        self.executable = executable
        self.extra_environment = dict(extra_environment or {})
        self.provider = provider
        project_root = Path(__file__).resolve().parents[3]
        self.patch_manifest = Path(
            patch_manifest
            or project_root / "patches" / "opencode" / "patch-manifest.json"
        ).resolve()
        self.session_capture = session_capture
        self.sandbox_options = sandbox_options or AnchorSandboxOptions()
        self.probe_timeout_seconds = float(probe_timeout_seconds)
        self._attestation: BinaryAttestation | None = None

    @property
    def backend_name(self) -> str:
        if self.provider == DEFAULT_PROVIDER:
            return "opencode-kimi-anchor-sandbox"
        return f"opencode-{self.provider.provider_id}-anchor-sandbox"

    def api_key_present(self) -> bool:
        value = self.extra_environment.get(
            self.provider.key_env, os.environ.get(self.provider.key_env, "")
        )
        return bool(value.strip())

    def _selected_credential_fingerprint(self) -> tuple[int, str] | None:
        value = self.extra_environment.get(
            self.provider.key_env, os.environ.get(self.provider.key_env, "")
        )
        if not value:
            return None
        return credential_fingerprint(value)

    def _resolved_executable(self) -> str | None:
        if os.name == "nt" and not Path(self.executable).suffix:
            command_shim = shutil.which(f"{self.executable}.cmd")
            if command_shim:
                return command_shim
        return shutil.which(self.executable)

    def available(self) -> bool:
        return self._resolved_executable() is not None

    @staticmethod
    def is_unforced_agent_config(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        options = value.get("options")
        return "requireInitialToolCall" not in value and not (
            isinstance(options, dict) and "requireInitialToolCall" in options
        )

    def probe_patched(self, config_path: Path) -> tuple[bool, str]:
        executable = self._resolved_executable()
        if executable is None:
            return False, "patched executable is missing"
        try:
            attestation = verify_binary_attestation(
                executable,
                patch_manifest=self.patch_manifest,
                linux_executable=self.sandbox_options.linux_executable,
            )
        except (OSError, ValueError) as error:
            return False, str(error)
        environment = self._environment(config_path)
        environment.pop(SANDBOX_API_KEY_ENV, None)
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
            if not self.is_unforced_agent_config(payload):
                return False, "resolved agent unexpectedly forces an initial tool call"
            verify_binary_identity(attestation)
            with tempfile.TemporaryDirectory(
                prefix="opencode-behavioral-probe-",
                dir=config_path.parent.parent,
                # Windows scanners can briefly retain a just-exited OpenCode
                # file handle. These directories contain only offline probe
                # fixtures, never credentials or task/sample content.
                ignore_cleanup_errors=True,
            ) as probe_root:
                passed, reason = run_behavioral_probe(
                    attestation.executable,
                    probe_root=Path(probe_root),
                    environment=environment,
                    timeout_seconds=self.probe_timeout_seconds,
                )
            if not passed:
                return False, reason
            if self.provider.is_responses:
                with tempfile.TemporaryDirectory(
                    prefix="opencode-responses-wire-probe-",
                    dir=config_path.parent.parent,
                    ignore_cleanup_errors=True,
                ) as probe_root:
                    passed, reason = run_responses_wire_probe(
                        attestation.executable,
                        probe_root=Path(probe_root),
                        environment=environment,
                        provider=self.provider,
                        timeout_seconds=self.probe_timeout_seconds,
                    )
                if not passed:
                    return False, reason
            self._attestation = attestation.with_behavioral_probe()
            return True, reason
        except subprocess.TimeoutExpired:
            return False, "patched capability probe timed out"
        finally:
            shutil.rmtree(config_path.parent / "runtime", ignore_errors=True)

    def _environment(self, config_path: Path) -> dict[str, str]:
        runtime_root = config_path.parent / "runtime"
        config_root = runtime_root / "config"
        data_root = runtime_root / "data"
        cache_root = runtime_root / "cache"
        for directory in (config_root, data_root, cache_root):
            directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(self.extra_environment)
        source_key = environment.get(self.provider.key_env, "")
        # Do not forward the provider-specific host variable.  The patched
        # sandbox consumes a one-child alias, creates a Podman secret from it,
        # and scrubs it before launching any other process.
        environment.pop(self.provider.key_env, None)
        environment.pop(SANDBOX_API_KEY_ENV, None)
        if source_key:
            environment[SANDBOX_API_KEY_ENV] = source_key
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

    def command(
        self, *, sample_id: str, prompt: str, workspace: Path, config_path: Path
    ) -> list[str]:
        executable = self._verified_executable()
        run_id = safe_sample_id(sample_id)
        return [
            str(executable),
            "anchor",
            "run",
            "--run-id",
            run_id,
            "--workspace",
            str(workspace.resolve()),
            "--config",
            str(config_path.resolve()),
            *self.sandbox_options.command_options(),
            "--model",
            f"{self.provider.provider_id}/{self.provider.model}",
            "--agent",
            AGENT_ID,
            "--variant",
            self.provider.variant,
            "--title",
            f"anchor-distiller:{run_id}",
            prompt,
        ]

    def _export_command(
        self, *, sample_id: str, session_id: str, workspace: Path, config_path: Path
    ) -> list[str]:
        executable = self._verified_executable()
        run_id = safe_sample_id(sample_id)
        return [
            str(executable),
            "anchor",
            "export",
            "--run-id",
            run_id,
            "--workspace",
            str(workspace.resolve()),
            "--config",
            str(config_path.resolve()),
            *self.sandbox_options.command_options(),
            "--session",
            session_id,
        ]

    def _cleanup_command(self, *, sample_id: str, workspace: Path) -> list[str]:
        executable = self._verified_executable()
        run_id = safe_sample_id(sample_id)
        return [
            str(executable),
            "anchor",
            "cleanup",
            "--run-id",
            run_id,
            "--workspace",
            str(workspace.resolve()),
            *self.sandbox_options.cleanup_command_options(),
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
        execution_config = config_path
        verify_launch_identity(self._attestation)  # type: ignore[arg-type]
        try:
            process = subprocess.Popen(
                self.command(
                    sample_id=sample_id,
                    prompt=prompt,
                    workspace=workspace,
                    config_path=execution_config,
                ),
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
        except OSError:
            cleanup_code = self._cleanup_sandbox(
                sample_id=sample_id, workspace=workspace
            )
            launch_errors: tuple[str, ...] = ("anchor_sandbox_launch_failed",)
            if cleanup_code is not None:
                launch_errors += (cleanup_code,)
            attestation = self._attestation
            return AgentExecution(
                exit_code=127,
                timed_out=False,
                duration_ms=(time.perf_counter() - started) * 1000,
                error_codes=launch_errors,
                isolated_runtime_path=str(config_path.parent / "runtime"),
                opencode_version=attestation.opencode_version
                if attestation is not None
                else None,
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
        public_outcome = parse_public_outcome(stdout)
        error_codes = list(classify_error_metadata(stdout, stderr))
        runtime_path = config_path.parent / "runtime"
        session_id = _extract_session_id(stdout)
        export_path: Path | None = None
        if session_id is not None:
            try:
                exported = subprocess.run(
                    self._export_command(
                        sample_id=sample_id,
                        session_id=session_id,
                        workspace=workspace,
                        config_path=execution_config,
                    ),
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
                    error_codes.append("controlled_session_export_failed")
            except (OSError, subprocess.TimeoutExpired, ValueError):
                error_codes.append("controlled_session_export_failed")
        else:
            error_codes.append("controlled_session_id_missing")
        if timed_out:
            error_codes.append("wrapper_timeout")
        return AgentExecution(
            exit_code=process.returncode if process.returncode is not None else 124,
            timed_out=timed_out,
            duration_ms=duration_ms,
            trace=trace,
            stdout_sha256=digest_text(stdout),
            stderr_sha256=digest_text(stderr),
            rejected_events=rejected,
            error_codes=tuple(dict.fromkeys(error_codes)),
            public_outcome=public_outcome,
            controlled_session_id=session_id,
            controlled_export_path=str(export_path) if export_path else None,
            isolated_runtime_path=str(runtime_path),
            opencode_version=(
                self._attestation.opencode_version
                if self._attestation is not None
                else None
            ),
        )

    def _cleanup_sandbox(self, *, sample_id: str, workspace: Path) -> str | None:
        try:
            completed = subprocess.run(
                self._cleanup_command(sample_id=sample_id, workspace=workspace),
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return "anchor_sandbox_cleanup_failed"
        return None if completed.returncode == 0 else "anchor_sandbox_cleanup_failed"

    def cleanup_sandbox(self, *, sample_id: str, workspace: Path) -> None:
        """Explicitly reap any deterministic leftover before final receipt issue."""

        code = self._cleanup_sandbox(sample_id=sample_id, workspace=workspace)
        if code is not None:
            raise RuntimeError(code)

    def finalize_capture(
        self,
        *,
        execution: AgentExecution,
        sample_id: str,
        workspace: Path,
        validators: tuple[dict[str, object], ...],
        final_diff: tuple[dict[str, object], ...] = (),
        skill_provenance: tuple[SkillProvenance, ...] = (),
    ) -> tuple[bool, str | None]:
        runtime = (
            Path(execution.isolated_runtime_path)
            if execution.isolated_runtime_path
            else None
        )
        export_path = (
            Path(execution.controlled_export_path)
            if execution.controlled_export_path
            else None
        )
        export_bytes = (
            export_path.read_bytes() if export_path and export_path.is_file() else b""
        )
        candidate: dict[str, object] | None = None
        staged_candidate: dict[str, object] | None = None
        failure_code: str | None = None
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
                "public_outcome": _public_outcome_capture(outcome),
                "final_diff": list(final_diff),
                "skill_provenance": [asdict(item) for item in skill_provenance],
                "quality": {
                    "agent_exit_code": execution.exit_code,
                    "timed_out": execution.timed_out,
                    "rejected_events": execution.rejected_events,
                    "error_codes": list(execution.error_codes),
                },
            }
            conversion_policy = SessionConversionPolicy(
                workspace_root=workspace.resolve(),
                heldout_cases=self.session_capture.heldout_cases,
                heldout_fixtures_root=self.session_capture.heldout_fixtures_root,
                heldout_manifest=self.session_capture.heldout_manifest,
                selected_credential_fingerprint=(
                    self._selected_credential_fingerprint()
                ),
            )
            if self.session_capture.mode == "collect":
                staged_candidate = convert_controlled_session_staging(
                    export_data, capture, conversion_policy
                )
            else:
                candidate = convert_controlled_session(
                    export_data, capture, conversion_policy
                )
        except (OSError, ValueError) as error:
            failure_code = (
                error.code if isinstance(error, QuarantineError) else "conversion_error"
            )
        finally:
            cleanup_code = (
                None
                if "anchor_sandbox_launch_failed" in execution.error_codes
                else self._cleanup_sandbox(sample_id=sample_id, workspace=workspace)
            )
            if cleanup_code is not None:
                failure_code = cleanup_code
            if runtime is not None:
                shutil.rmtree(runtime, ignore_errors=True)
        if failure_code is not None or (
            self.session_capture is not None
            and self.session_capture.mode == "strict"
            and candidate is None
        ):
            code = failure_code or "conversion_error"
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
            prefix = (
                "session_hard_reject_"
                if self.session_capture and self.session_capture.mode == "collect"
                else ""
            )
            return False, prefix + code
        assert self.session_capture is not None
        with _CAPTURE_LOCK:
            if staged_candidate is not None:
                assert self.session_capture.staging_path is not None
                append_jsonl(self.session_capture.staging_path, staged_candidate)
            if candidate is not None:
                append_jsonl(self.session_capture.candidates_path, candidate)
        return True, None


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
