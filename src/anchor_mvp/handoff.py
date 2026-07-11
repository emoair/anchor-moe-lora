"""Fail-closed distillation-to-training handoff orchestration.

The parent process is the only credential owner.  Live credentials are read with
``getpass`` and supplied to a short-lived child through its environment.  They are
never accepted as command-line/config values and distillation child output is reduced
to exit codes and hashes before it is persisted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
from uuid import uuid4

import yaml

from .benchmark.heldout import check_training_leakage, file_sha256
from .tooling.session_export import CANDIDATE_SCHEMA_VERSION, SECRET_PATTERNS
from .training.config import load_training_config, select_adapter
from .training.schema import validate_jsonl


CONFIG_SCHEMA = "anchor.distill-train-handoff.config.v1"
STATUS_SCHEMA = "anchor.distill-train-handoff.status.v1"
MANIFEST_SCHEMA = "anchor.training-handoff.v1"
EXPERTS = ("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate")
RAMP = (1, 2, 4, 8)
SAFE_HANDOFF_TRIGGERS = frozenset({"provider_quota_exhausted", "automation_complete"})
_FORBIDDEN_SECRET_KEYS = frozenset(
    {"api_key", "apikey", "secret", "token", "password", "authorization"}
)
_CREDENTIAL_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|AUTHORIZATION|CREDENTIAL)(?:$|_)",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _walk_secret_config(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.casefold() in _FORBIDDEN_SECRET_KEYS:
                raise ValueError(
                    "handoff config must contain credential environment names only; "
                    f"secret-valued field is forbidden: {'.'.join((*path, key))}"
                )
            _walk_secret_config(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_secret_config(item, (*path, str(index)))
    elif isinstance(value, str) and any(pattern.search(value) for pattern in SECRET_PATTERNS):
        raise ValueError(
            "handoff config appears to contain credential material: "
            + (".".join(path) or "<root>")
        )


def _project_path(root: Path, value: object, label: str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root") from exc
    return resolved


class HandoffConfig:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        loaded = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or loaded.get("schema_version") != CONFIG_SCHEMA:
            raise ValueError("unsupported distill/train handoff config schema")
        _walk_secret_config(loaded)
        self.raw = dict(loaded)
        root_value = self.raw.get("project_root", "../..")
        candidate_root = (self.path.parent / str(root_value)).resolve()
        if not candidate_root.is_dir():
            raise ValueError("project_root is missing")
        self.root = candidate_root
        self.config_sha256 = _sha256_bytes(_canonical_bytes(self.raw))

        self.execution = self._section("execution")
        self.distillation = self._section("distillation")
        self.snapshot = self._section("snapshot")
        self.training = self._section("training")
        max_concurrency = int(self.execution.get("max_concurrency", 8))
        if max_concurrency not in RAMP:
            raise ValueError("execution.max_concurrency must be one of 1,2,4,8")
        self.ramp = tuple(value for value in RAMP if value <= max_concurrency)
        if int(self.training.get("max_parallel_gpu_jobs", 1)) != 1:
            raise ValueError("training.max_parallel_gpu_jobs must be exactly 1")
        trigger = str(
            self.training.get("handoff_trigger", "provider_quota_exhausted")
        )
        if trigger not in SAFE_HANDOFF_TRIGGERS:
            raise ValueError(
                "training.handoff_trigger must be provider_quota_exhausted or automation_complete"
            )

    def _section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        return dict(value)

    def path_value(self, section: Mapping[str, Any], name: str) -> Path:
        if name not in section:
            raise ValueError(f"missing configured path: {name}")
        return _project_path(self.root, section[name], name)

    @property
    def state_dir(self) -> Path:
        return _project_path(
            self.root, self.raw.get("state_dir", "runs/distill-train-handoff"), "state_dir"
        )


class StatusStore:
    def __init__(self, config: HandoffConfig) -> None:
        self.path = config.state_dir / "status.json"
        self.events_path = config.state_dir / "events.jsonl"
        if self.path.is_file():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("schema_version") != STATUS_SCHEMA:
                raise ValueError("unsupported handoff status schema")
            if value.get("config_sha256") != config.config_sha256:
                raise ValueError("handoff config changed; use a new state_dir")
            self.value = value
        else:
            self.value = {
                "schema_version": STATUS_SCHEMA,
                "run_id": uuid4().hex,
                "config_sha256": config.config_sha256,
                "mode": None,
                "phase": "ready",
                "created_at": _now(),
                "updated_at": _now(),
                "event_sequence": 0,
                "execution": {
                    "completed_concurrency": [],
                    "accepted_gold_count": 0,
                    "session_candidate_count": 0,
                    "single_converter_gate_passed": False,
                },
                "distillation": {
                    "cycles": 0,
                    "quota_epoch_id": None,
                    "terminal_state": None,
                    "classification": None,
                },
                "snapshot": None,
                "handoff": None,
                "training": {"active_job": None, "jobs": []},
            }
            self.save()

    def save(self) -> None:
        self.value["updated_at"] = _now()
        _atomic_json(self.path, self.value)

    def event(self, kind: str, **data: Any) -> None:
        self.value["event_sequence"] = int(self.value["event_sequence"]) + 1
        event = {
            "schema_version": STATUS_SCHEMA,
            "run_id": self.value["run_id"],
            "sequence": self.value["event_sequence"],
            "time": _now(),
            "type": kind,
            "data": data,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.save()


def classify_distillation_terminal(status: Mapping[str, Any]) -> str:
    """Classify only durable automation states; never parse arbitrary HTTP prose."""

    state = str(status.get("state", "unknown"))
    return {
        "provider_quota_exhausted": "provider_quota_exhausted",
        "complete": "automation_complete",
        "cooldown": "temporary_rate_limit",
        "client_deadline": "client_deadline",
        "budget_exhausted": "local_safety_budget",
        "failed": "failed",
        "gate_blocked": "data_gate_blocked",
        "running": "nonterminal",
        "ready": "nonterminal",
    }.get(state, "unknown")


def inspect_execution_artifacts(
    accepted_gold: Path, session_candidates: Path, *, minimum_gold: int
) -> dict[str, Any]:
    errors: list[str] = []
    accepted_ids: set[str] = set()
    candidate_ids: set[str] = set()
    if not accepted_gold.is_file():
        errors.append("accepted_gold_missing")
    else:
        for line_number, line in enumerate(
            accepted_gold.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"accepted_gold_invalid_json:{line_number}")
                continue
            outcome = record.get("public_outcome") if isinstance(record, Mapping) else None
            sample_id = str(record.get("sample_id", "")) if isinstance(record, Mapping) else ""
            if (
                not isinstance(record, Mapping)
                or record.get("schema_version") != "anchor.tool-gold.v1"
                or record.get("success") is not True
                or not isinstance(outcome, Mapping)
                or outcome.get("status") != "completed"
                or not sample_id
            ):
                errors.append(f"accepted_gold_rejected:{line_number}")
                continue
            if sample_id in accepted_ids:
                errors.append(f"accepted_gold_duplicate:{sample_id}")
            accepted_ids.add(sample_id)

    if not session_candidates.is_file():
        errors.append("session_candidates_missing")
    else:
        for line_number, line in enumerate(
            session_candidates.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"session_candidate_invalid_json:{line_number}")
                continue
            sample_id = str(record.get("sample_id", "")) if isinstance(record, Mapping) else ""
            trajectory = record.get("trajectory") if isinstance(record, Mapping) else None
            validators = record.get("validators") if isinstance(record, Mapping) else None
            outcome = record.get("public_outcome") if isinstance(record, Mapping) else None
            calls: dict[str, str] = {}
            results: dict[str, str] = {}
            if isinstance(trajectory, list):
                for item in trajectory:
                    if not isinstance(item, Mapping):
                        continue
                    if item.get("type") == "tool_call":
                        calls[str(item.get("call_id"))] = str(item.get("tool"))
                    elif item.get("type") == "tool_result":
                        results[str(item.get("call_id"))] = str(item.get("tool"))
            valid = bool(
                isinstance(record, Mapping)
                and record.get("schema_version") == CANDIDATE_SCHEMA_VERSION
                and sample_id
                and calls
                and calls == results
                and isinstance(validators, list)
                and validators
                and all(
                    isinstance(item, Mapping)
                    and item.get("status") == "PASS"
                    and item.get("exit_code") == 0
                    for item in validators
                )
                and isinstance(outcome, Mapping)
                and outcome.get("status") == "completed"
            )
            if not valid:
                errors.append(f"session_candidate_rejected:{line_number}")
                continue
            if sample_id in candidate_ids:
                errors.append(f"session_candidate_duplicate:{sample_id}")
            candidate_ids.add(sample_id)

    overlap = accepted_ids & candidate_ids
    if len(accepted_ids) < minimum_gold:
        errors.append("accepted_gold_below_minimum")
    if not overlap:
        errors.append("gold_session_sample_mismatch")
    return {
        "passed": not errors,
        "accepted_gold_count": len(accepted_ids),
        "session_candidate_count": len(candidate_ids),
        "matched_sample_count": len(overlap),
        "accepted_gold_sha256": file_sha256(accepted_gold) if accepted_gold.is_file() else None,
        "session_candidates_sha256": (
            file_sha256(session_candidates) if session_candidates.is_file() else None
        ),
        "errors": errors,
    }


def _secret_scan_jsonl(path: Path) -> list[str]:
    findings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            findings.append(f"{path.name}:{line_number}")
    return findings


def evaluate_snapshot(
    *,
    datasets: Mapping[str, Path],
    minimum_records: Mapping[str, int],
    heldout_cases: Path,
    heldout_fixtures_root: Path,
    heldout_manifest: Path,
    sop_sources: Iterable[Path] = (),
    execution_gate: Mapping[str, Any],
    ramp_complete: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    all_ids: list[str] = []
    secret_findings: list[str] = []
    errors: list[str] = []
    for expert in EXPERTS:
        path = datasets.get(expert)
        if path is None or not path.is_file():
            reports[expert] = {"exists": False, "ok": False}
            errors.append(f"dataset_missing:{expert}")
            continue
        try:
            report = validate_jsonl(path, allowed_experts=[expert])
            count = int(report["valid_records"])
            if count < int(minimum_records[expert]):
                errors.append(f"minimum_records:{expert}")
            ids: list[str] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        ids.append(str(json.loads(line)["id"]))
            all_ids.extend(ids)
            secret_findings.extend(_secret_scan_jsonl(path))
            reports[expert] = {
                **report,
                "exists": True,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "minimum_required": int(minimum_records[expert]),
            }
        except (OSError, ValueError) as exc:
            reports[expert] = {
                "exists": True,
                "ok": False,
                "error_type": type(exc).__name__,
            }
            errors.append(f"schema:{expert}")
    duplicates = len(all_ids) - len(set(all_ids))
    if duplicates:
        errors.append("duplicate_ids")
    if secret_findings:
        errors.append("secret_scan")
    try:
        leakage = check_training_leakage(
            heldout_cases,
            heldout_fixtures_root,
            heldout_manifest,
            datasets.values(),
            sop_sources,
        )
        if leakage.get("status") != "PASS" or leakage.get("collision_count") != 0:
            errors.append("heldout_leakage")
    except (OSError, ValueError) as exc:
        leakage = {"status": "ERROR", "error_type": type(exc).__name__}
        errors.append("heldout_gate_error")
    if not execution_gate.get("passed"):
        errors.append("execution_artifacts")
    if not ramp_complete:
        errors.append("execution_ramp_incomplete")

    file_bindings = [
        {"expert": expert, "path": str(datasets[expert]), "sha256": reports[expert].get("sha256")}
        for expert in EXPERTS
        if expert in datasets and expert in reports
    ]
    snapshot_sha = _sha256_bytes(_canonical_bytes(file_bindings))
    return {
        "passed": not errors,
        "snapshot_sha256": snapshot_sha,
        "datasets": reports,
        "cross_file_duplicate_ids": duplicates,
        "secret_findings": secret_findings,
        "heldout": leakage,
        "execution": dict(execution_gate),
        "execution_ramp_complete": ramp_complete,
        "errors": errors,
    }


def inspect_lowmem_training(
    config_path: Path,
    adapters: Sequence[str],
    *,
    dataset_bindings: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    config = load_training_config(config_path)
    training = config["training"]
    if training.get("runtime_engine") != "manual_active_labels_v2":
        errors.append("runtime_engine")
    if training.get("gradient_checkpointing") is not True:
        errors.append("gradient_checkpointing")
    if int(training.get("per_device_train_batch_size", 0)) != 1:
        errors.append("batch_size")
    peak = training.get("maximum_training_peak_vram_gib")
    if not isinstance(peak, (int, float)) or float(peak) > 9.0:
        errors.append("maximum_training_peak_vram_gib")
    for adapter in adapters:
        try:
            selected = select_adapter(config, adapter, None)
        except ValueError:
            errors.append(f"adapter:{adapter}")
            continue
        if dataset_bindings is not None and adapter in dataset_bindings:
            config_root = (
                Path(str(config["_config_path"])).parent
                / str(config.get("paths", {}).get("project_root", "../.."))
            ).resolve()
            configured = {
                (config_root / str(item)).resolve()
                for item in selected["active_adapter"]["datasets"]
            }
            if configured != {dataset_bindings[adapter].resolve()}:
                errors.append(f"dataset_binding:{adapter}")
    return {
        "passed": not errors,
        "runtime_engine": training.get("runtime_engine"),
        "maximum_training_peak_vram_gib": peak,
        "per_device_train_batch_size": training.get("per_device_train_batch_size"),
        "gradient_checkpointing": training.get("gradient_checkpointing"),
        "errors": errors,
    }


def write_handoff_manifest(
    config: HandoffConfig,
    status: StatusStore,
    snapshot: Mapping[str, Any],
    execution_gate: Mapping[str, Any],
    lowmem_gate: Mapping[str, Any],
) -> tuple[Path, str]:
    if not snapshot.get("passed") or not execution_gate.get("passed") or not lowmem_gate.get("passed"):
        raise ValueError("refusing handoff manifest because one or more gates failed")
    path = config.path_value(config.training, "handoff_manifest")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": _now(),
        "run_id": status.value["run_id"],
        "config_sha256": config.config_sha256,
        "trigger": status.value["distillation"]["classification"],
        "quota_epoch_id": status.value["distillation"]["quota_epoch_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "snapshot": dict(snapshot),
        "execution_gate": dict(execution_gate),
        "training_profile_gate": dict(lowmem_gate),
        "training": {
            "config": str(config.path_value(config.training, "config")),
            "adapters": list(config.training.get("adapters", EXPERTS)),
            "execution": "sequential_single_lora",
            "max_parallel_gpu_jobs": 1,
        },
    }
    _atomic_json(path, manifest)
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return path, digest


def verify_frozen_handoff(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("training handoff manifest or SHA sidecar is missing")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    if file_sha256(path) != expected:
        raise ValueError("training handoff manifest changed after freeze")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported training handoff manifest")
    for report in value["snapshot"]["datasets"].values():
        path_value = Path(str(report["path"]))
        if not path_value.is_file() or file_sha256(path_value) != report.get("sha256"):
            raise ValueError("dataset changed after training handoff freeze")
    return value


def _safe_child_environment(extra: Mapping[str, str], *, strip: Iterable[str] = ()) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not _CREDENTIAL_ENV_NAME.search(name)
    }
    for name in strip:
        environment.pop(name, None)
    environment.update(extra)
    return environment


@contextmanager
def _credential_free_parent_environment() -> Iterator[None]:
    """Temporarily hide inherited credentials from OpenCode's env-copying executor."""

    removed = {
        name: value
        for name, value in tuple(os.environ.items())
        if _CREDENTIAL_ENV_NAME.search(name)
    }
    try:
        for name in removed:
            os.environ.pop(name, None)
        yield
    finally:
        os.environ.update(removed)


def _run_child(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    secrets: Sequence[str] = (),
    log_path: Path | None = None,
) -> dict[str, Any]:
    child_environment = dict(environment)
    source_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        source_root + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else source_root
    )
    completed = subprocess.run(
        list(command),
        cwd=Path(__file__).resolve().parents[2],
        env=child_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    if any(secret and secret.encode("utf-8") in combined for secret in secrets):
        raise RuntimeError("child output contained a credential; output was discarded")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(completed.stdout + b"\n--- stderr ---\n" + completed.stderr)
    return {
        "returncode": completed.returncode,
        "stdout_sha256": _sha256_bytes(completed.stdout),
        "stderr_sha256": _sha256_bytes(completed.stderr),
        "stdout": completed.stdout,
    }


@contextmanager
def _gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("another handoff-owned GPU training job may be active") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _read_credential(name: str) -> str:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", name):
        raise ValueError("credential environment name is invalid")
    value = getpass.getpass(f"{name} (masked, process memory only): ")
    if not value:
        raise ValueError(f"empty credential: {name}")
    return value


def _run_opencode_stage(
    config: HandoffConfig, *, concurrency: int, credential_env: str, credential: str
) -> dict[str, Any]:
    """Run exactly one audited ramp stage using the existing batch implementation."""

    from .tooling import (
        LiveBatchConfig,
        OpenCodeExecutor,
        SkillSourceRegistry,
        ToolPolicy,
        batch_run_succeeded,
        load_candidate_samples,
        persist_attempts_and_gold,
        run_live_batch,
        verify_execution_split,
        write_opencode_config,
    )

    batch_path = config.path_value(config.execution, "batch_config")
    batch = LiveBatchConfig.load(config.root, batch_path)
    stage_index = batch.concurrency_stages.index(concurrency)
    offset = sum(batch.samples_per_stage[:stage_index])
    count = batch.samples_per_stage[stage_index]
    registry = SkillSourceRegistry(config.root, batch.skill_registry)
    heldout_ids, heldout_requirements = verify_execution_split(
        config.root, batch.split_policy, batch.candidate_manifest
    )
    all_samples = load_candidate_samples(
        config.root,
        batch.candidate_manifest,
        registry,
        heldout_identifiers=heldout_ids,
        heldout_requirements=heldout_requirements,
    )
    selected = all_samples[offset : offset + count]
    if len(selected) != count:
        raise ValueError(f"OpenCode stage {concurrency} lacks audited candidates")
    stage_config = replace(
        batch,
        concurrency_stages=(concurrency,),
        samples_per_stage=(count,),
    )
    if batch.opencode_executable is None:
        raise ValueError("batch config requires a patched OpenCode executable")
    with _credential_free_parent_environment():
        executor = OpenCodeExecutor(
            executable=str(batch.opencode_executable),
            extra_environment={credential_env: credential},
            session_capture=batch.controlled_capture(),
        )
        with tempfile.TemporaryDirectory(
            prefix="handoff-opencode-probe-", dir=config.state_dir
        ) as raw:
            policy = ToolPolicy(
                max_iterations=batch.max_iterations,
                timeout_seconds=batch.timeout_seconds,
            )
            probe_config = write_opencode_config(Path(raw) / "opencode.json", policy)
            patched, reason = executor.probe_patched(probe_config)
        if not patched:
            raise RuntimeError(f"patched OpenCode capability gate failed: {reason}")
        stages = run_live_batch(
            samples=selected,
            config=stage_config,
            executor=executor,
            max_stages=1,
            on_stage=lambda records: persist_attempts_and_gold(
                records,
                attempts_path=batch.attempts_output,
                gold_path=batch.gold_output,
            ),
        )
    return {
        "passed": batch_run_succeeded(stages, 1),
        "concurrency": concurrency,
        "records": len(stages[0].records) if stages else 0,
        "accepted": sum(record.success for record in stages[0].records) if stages else 0,
    }


def _convert_configured_single_session(config: HandoffConfig) -> dict[str, Any]:
    """Convert only a configured controlled export; arbitrary history is never scanned."""

    from .tooling.session_export import (
        QuarantineError,
        SessionConversionPolicy,
        append_jsonl,
        convert_controlled_session,
        quarantine_record,
    )

    required = ("raw_export", "capture", "workspace", "session_quarantine")
    if not all(config.execution.get(name) for name in required):
        return {"converted": False, "reason": "controlled_export_not_configured"}
    raw_path = config.path_value(config.execution, "raw_export")
    capture_path = config.path_value(config.execution, "capture")
    if not raw_path.is_file() or not capture_path.is_file():
        return {"converted": False, "reason": "controlled_export_not_ready"}
    export_bytes = raw_path.read_bytes()
    capture: object = None
    try:
        export_data = json.loads(export_bytes.decode("utf-8"))
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if not isinstance(export_data, dict) or not isinstance(capture, dict):
            raise QuarantineError("input_not_object")
        candidate = convert_controlled_session(
            export_data,
            capture,
            SessionConversionPolicy(
                workspace_root=config.path_value(config.execution, "workspace"),
                heldout_cases=config.path_value(config.snapshot, "heldout_cases"),
                heldout_fixtures_root=config.path_value(
                    config.snapshot, "heldout_fixtures_root"
                ),
                heldout_manifest=config.path_value(config.snapshot, "heldout_manifest"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        error = QuarantineError("invalid_json_or_encoding")
    except (OSError, ValueError) as caught:
        error = caught if isinstance(caught, QuarantineError) else QuarantineError(
            "conversion_error"
        )
    else:
        append_jsonl(
            config.path_value(config.execution, "session_candidates"), candidate
        )
        return {"converted": True, "sample_id": candidate["sample_id"]}
    sample_id = str(capture.get("sample_id")) if isinstance(capture, dict) else None
    append_jsonl(
        config.path_value(config.execution, "session_quarantine"),
        quarantine_record(
            sample_id=sample_id, code=error.code, export_bytes=export_bytes
        ),
    )
    return {"converted": False, "reason": error.code}


def _datasets_and_minimums(config: HandoffConfig) -> tuple[dict[str, Path], dict[str, int]]:
    raw_datasets = config.snapshot.get("datasets")
    if not isinstance(raw_datasets, Mapping) or set(raw_datasets) != set(EXPERTS):
        raise ValueError("snapshot.datasets must map exactly the five experts")
    datasets = {
        expert: _project_path(config.root, raw_datasets[expert], f"dataset.{expert}")
        for expert in EXPERTS
    }
    raw_min = config.snapshot.get("minimum_records_per_expert", 1)
    if isinstance(raw_min, Mapping):
        minimums = {expert: int(raw_min[expert]) for expert in EXPERTS}
    else:
        minimums = {expert: int(raw_min) for expert in EXPERTS}
    if any(value < 1 for value in minimums.values()):
        raise ValueError("minimum records must be positive")
    return datasets, minimums


def _run_training_jobs(
    config: HandoffConfig,
    status: StatusStore,
    *,
    confirm_training: bool,
) -> None:
    if not confirm_training:
        status.value["phase"] = "handoff_ready"
        status.event("training_not_confirmed")
        return
    handoff_path = Path(status.value["handoff"]["path"])
    verify_frozen_handoff(handoff_path)
    training_config = config.path_value(config.training, "config")
    adapters = tuple(str(item) for item in config.training.get("adapters", EXPERTS))
    logs_dir = config.path_value(config.training, "logs_dir")
    known_credentials = {
        str(config.execution.get("credential_env", "KIMI_CODE_API_KEY")),
        str(config.distillation.get("credential_env", "KIMI_API_KEY")),
    }
    environment = _safe_child_environment({}, strip=known_credentials)
    preflight = _run_child(
        [
            sys.executable,
            "-m",
            "anchor_mvp.training",
            "preflight",
            "--config",
            str(training_config),
            "--dry-run",
        ],
        environment=environment,
        log_path=logs_dir / "preflight.log",
    )
    if preflight["returncode"] != 0:
        status.value["phase"] = "training_preflight_blocked"
        status.event("training_preflight_blocked", returncode=preflight["returncode"])
        return
    completed = {
        item["adapter"] for item in status.value["training"]["jobs"] if item["state"] == "complete"
    }
    lock_path = config.state_dir / "gpu-job.lock"
    with _gpu_lock(lock_path):
        for adapter in adapters:
            if adapter in completed:
                continue
            verify_frozen_handoff(handoff_path)
            status.value["phase"] = "training"
            status.value["training"]["active_job"] = adapter
            status.event("training_job_started", adapter=adapter)
            result = _run_child(
                [
                    sys.executable,
                    "-m",
                    "anchor_mvp.training",
                    "train",
                    "--config",
                    str(training_config),
                    "--adapter",
                    adapter,
                    "--execute",
                ],
                environment=environment,
                log_path=logs_dir / f"{adapter}.log",
            )
            job = {
                "adapter": adapter,
                "state": "complete" if result["returncode"] == 0 else "failed",
                "returncode": result["returncode"],
                "stdout_sha256": result["stdout_sha256"],
                "stderr_sha256": result["stderr_sha256"],
                "completed_at": _now(),
            }
            status.value["training"]["jobs"].append(job)
            status.value["training"]["active_job"] = None
            status.event("training_job_finished", adapter=adapter, state=job["state"])
            if result["returncode"] != 0:
                status.value["phase"] = "training_failed"
                status.save()
                return
    status.value["phase"] = "complete"
    status.event("training_completed", adapters=list(adapters))


def run_coordinator(
    config: HandoffConfig,
    *,
    confirm_live: bool,
    confirm_training: bool,
) -> dict[str, Any]:
    status = StatusStore(config)
    dry_run = not confirm_live
    status.value["mode"] = "dry-run" if dry_run else "live"
    status.event("coordinator_started", mode=status.value["mode"])

    accepted_gold = config.path_value(config.execution, "accepted_gold")
    session_candidates = config.path_value(config.execution, "session_candidates")
    execution_gate = inspect_execution_artifacts(
        accepted_gold,
        session_candidates,
        minimum_gold=int(config.execution.get("minimum_accepted_gold", 1)),
    )
    status.value["execution"].update(
        {
            "accepted_gold_count": execution_gate["accepted_gold_count"],
            "session_candidate_count": execution_gate["session_candidate_count"],
            "single_converter_gate_passed": execution_gate["passed"],
        }
    )
    if dry_run:
        if execution_gate["passed"]:
            status.value["execution"]["completed_concurrency"] = list(config.ramp)
    else:
        completed = [
            int(item) for item in status.value["execution"]["completed_concurrency"]
        ]
        # Matching accepted gold + converted tool-call/results proves the single
        # execution gate already passed in an earlier idempotent run.
        if execution_gate["passed"] and not completed:
            completed.append(1)
            status.value["execution"]["completed_concurrency"] = completed
            status.event("single_execution_gate_recovered_from_artifacts")
        credential_env = str(
            config.execution.get("credential_env", "KIMI_CODE_API_KEY")
        )
        execution_credential: str | None = None
        for concurrency in config.ramp:
            if concurrency in completed:
                continue
            if execution_credential is None:
                execution_credential = _read_credential(credential_env)
            stage = _run_opencode_stage(
                config,
                concurrency=concurrency,
                credential_env=credential_env,
                credential=execution_credential,
            )
            status.event("opencode_stage_finished", **stage)
            if not stage["passed"]:
                status.value["phase"] = "execution_gate_blocked"
                status.save()
                return status.value
            completed.append(concurrency)
            status.value["execution"]["completed_concurrency"] = completed
            if concurrency == 1:
                conversion = _convert_configured_single_session(config)
                status.event("single_session_conversion", **conversion)
            execution_gate = inspect_execution_artifacts(
                accepted_gold,
                session_candidates,
                minimum_gold=int(config.execution.get("minimum_accepted_gold", 1)),
            )
            status.value["execution"].update(
                {
                    "accepted_gold_count": execution_gate["accepted_gold_count"],
                    "session_candidate_count": execution_gate[
                        "session_candidate_count"
                    ],
                    "single_converter_gate_passed": execution_gate["passed"],
                }
            )
            if not execution_gate["passed"]:
                status.value["phase"] = "execution_conversion_blocked"
                status.event(
                    "execution_conversion_blocked", errors=execution_gate["errors"]
                )
                return status.value

    ramp_complete = tuple(status.value["execution"]["completed_concurrency"]) == config.ramp
    if not execution_gate["passed"] or not ramp_complete:
        status.value["phase"] = "execution_gate_blocked"
        status.event(
            "execution_gate_blocked",
            errors=execution_gate["errors"],
            completed_concurrency=status.value["execution"]["completed_concurrency"],
        )
        return status.value

    if dry_run:
        terminal_state = str(
            config.distillation.get("dry_run_terminal_state", "provider_quota_exhausted")
        )
        automation_status: dict[str, Any] = {
            "state": terminal_state,
            "quota_epoch": {
                "epoch_id": str(config.distillation.get("dry_run_quota_epoch", "mock-epoch"))
            },
        }
    else:
        automation_config = config.path_value(config.distillation, "automation_config")
        credential_env = str(config.distillation.get("credential_env", "KIMI_API_KEY"))
        automation_status_path = config.path_value(config.distillation, "status_path")
        previous: dict[str, Any] | None = None
        if automation_status_path.is_file():
            loaded = json.loads(automation_status_path.read_text(encoding="utf-8"))
            previous = loaded if isinstance(loaded, dict) else None
        if previous is not None and classify_distillation_terminal(previous) in {
            "provider_quota_exhausted",
            "automation_complete",
        }:
            automation_status = previous
            status.event(
                "distillation_terminal_recovered",
                state=automation_status.get("state"),
            )
        else:
            teacher_credential = _read_credential(credential_env)
            result = _run_child(
                [
                    sys.executable,
                    "-m",
                    "anchor_mvp.data.automation",
                    "--config",
                    str(automation_config),
                ],
                environment=_safe_child_environment(
                    {credential_env: teacher_credential}
                ),
                secrets=(teacher_credential,),
            )
            status.value["distillation"]["cycles"] = (
                int(status.value["distillation"]["cycles"]) + 1
            )
            if not automation_status_path.is_file():
                status.value["phase"] = "distillation_failed"
                status.event(
                    "distillation_child_failed", returncode=result["returncode"]
                )
                return status.value
            automation_status = json.loads(
                automation_status_path.read_text(encoding="utf-8")
            )

    classification = classify_distillation_terminal(automation_status)
    epoch = automation_status.get("quota_epoch", {})
    status.value["distillation"].update(
        {
            "quota_epoch_id": epoch.get("epoch_id") if isinstance(epoch, Mapping) else None,
            "terminal_state": automation_status.get("state"),
            "classification": classification,
        }
    )
    status.event(
        "distillation_terminal_classified",
        state=automation_status.get("state"),
        classification=classification,
    )
    required_trigger = str(config.training.get("handoff_trigger", "provider_quota_exhausted"))
    if classification != required_trigger:
        status.value["phase"] = "distillation_stopped_no_handoff"
        status.event("handoff_not_triggered", required=required_trigger, observed=classification)
        return status.value

    datasets, minimums = _datasets_and_minimums(config)
    sop_sources = [
        _project_path(config.root, item, "sop_source")
        for item in config.snapshot.get("sop_sources", [])
    ]
    snapshot = evaluate_snapshot(
        datasets=datasets,
        minimum_records=minimums,
        heldout_cases=config.path_value(config.snapshot, "heldout_cases"),
        heldout_fixtures_root=config.path_value(config.snapshot, "heldout_fixtures_root"),
        heldout_manifest=config.path_value(config.snapshot, "heldout_manifest"),
        sop_sources=sop_sources,
        execution_gate=execution_gate,
        ramp_complete=ramp_complete,
    )
    status.value["snapshot"] = snapshot
    if not snapshot["passed"]:
        status.value["phase"] = "data_insufficient"
        status.event("data_snapshot_blocked", errors=snapshot["errors"])
        return status.value

    adapters = tuple(str(item) for item in config.training.get("adapters", EXPERTS))
    lowmem = inspect_lowmem_training(
        config.path_value(config.training, "config"),
        adapters,
        dataset_bindings=datasets,
    )
    if not lowmem["passed"]:
        status.value["phase"] = "training_profile_blocked"
        status.event("training_profile_blocked", errors=lowmem["errors"])
        return status.value
    existing_handoff = status.value.get("handoff")
    if isinstance(existing_handoff, Mapping):
        handoff_path = Path(str(existing_handoff.get("path", "")))
        frozen = verify_frozen_handoff(handoff_path)
        if (
            frozen.get("config_sha256") != config.config_sha256
            or frozen.get("snapshot_sha256") != snapshot["snapshot_sha256"]
            or frozen.get("trigger")
            != status.value["distillation"]["classification"]
        ):
            raise ValueError("existing training handoff does not match current frozen gates")
        handoff_sha = file_sha256(handoff_path)
        status.event(
            "training_handoff_recovered", path=str(handoff_path), sha256=handoff_sha
        )
    else:
        handoff_path, handoff_sha = write_handoff_manifest(
            config, status, snapshot, execution_gate, lowmem
        )
        status.value["handoff"] = {
            "path": str(handoff_path),
            "sha256": handoff_sha,
        }
        status.event(
            "training_handoff_frozen", path=str(handoff_path), sha256=handoff_sha
        )
    status.value["phase"] = "handoff_ready"
    if dry_run:
        status.event("dry_run_training_skipped")
        return status.value
    if bool(config.training.get("auto_start", False)):
        _run_training_jobs(config, status, confirm_training=confirm_training)
    return status.value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely hand off quota-exhausted distillation to sequential formal-v2 training"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="allow patched OpenCode/teacher API child processes; otherwise offline dry-run",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="allow sequential local GPU jobs after all handoff gates pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.confirm_training and not args.confirm_live:
        print(
            json.dumps({"ok": False, "error": "--confirm-training requires --confirm-live"}),
            file=sys.stderr,
        )
        return 2
    try:
        config = HandoffConfig(args.config)
        result = run_coordinator(
            config,
            confirm_live=bool(args.confirm_live),
            confirm_training=bool(args.confirm_training),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["phase"] in {"handoff_ready", "complete"} else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
