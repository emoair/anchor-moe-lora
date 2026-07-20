"""Crash-safe record checkpoints for the formal A--F benchmark.

The per-record journal is the source of truth.  ``records.raw.jsonl`` and
``status.json`` are atomically materialized projections, so an interrupted
write can never turn a partial JSON line into a completed benchmark record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import BaselineSpec, BenchmarkCase, BenchmarkRecord


CHECKPOINT_SCHEMA = "anchor.formal-af-checkpoint.v1"
RECORD_SCHEMA = "anchor.formal-af-checkpoint-record.v1"
STATUS_SCHEMA = "anchor.formal-af-progress.v1"


class FormalCheckpointError(RuntimeError):
    """A formal checkpoint is incomplete, corrupt, or bound to another run."""


@dataclass(frozen=True)
class FormalCheckpointBindings:
    config_sha256: str
    execution_contract_sha256: str
    run_manifest_sha256: str
    case_manifest_sha256: str
    leak_audit_sha256: str
    backend_identity: Mapping[str, Any]
    execution_options: Mapping[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _arm_label(spec: BaselineSpec) -> str:
    label = spec.group.split("_", 1)[0]
    if not label or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        for character in label
    ):
        raise FormalCheckpointError(f"invalid formal arm label for {spec.name!r}")
    return label


class FormalRunCheckpoint:
    """Persist and validate a prefix of completed formal arm/case records."""

    def __init__(
        self,
        destination: Path,
        *,
        manifest: dict[str, Any],
        specs: Sequence[BaselineSpec],
        cases: Sequence[BenchmarkCase],
        records: list[BenchmarkRecord],
        backend_label: str,
    ) -> None:
        self.destination = destination
        self.manifest = manifest
        self.specs = tuple(specs)
        self.cases = tuple(cases)
        self.records = records
        self._backend_label = backend_label
        self.manifest_digest = _digest(manifest)
        self._expected = [
            (case.case_id, spec.name, spec.group)
            for case in self.cases
            for spec in self.specs
        ]
        self._arm_by_baseline = {spec.name: _arm_label(spec) for spec in self.specs}
        self._record_dir = self.destination / ".formal-checkpoint" / "records"
        self._raw_path = self.destination / "records.raw.jsonl"
        self._status_path = self.destination / "status.json"

    @classmethod
    def open(
        cls,
        destination: str | Path,
        *,
        resume: bool,
        bindings: FormalCheckpointBindings,
        specs: Sequence[BaselineSpec],
        cases: Sequence[BenchmarkCase],
        backend_label: str,
    ) -> "FormalRunCheckpoint":
        root = Path(destination)
        if not specs or not cases:
            raise FormalCheckpointError("formal checkpoint requires non-empty arms and cases")
        for label, value in (
            ("config", bindings.config_sha256),
            ("execution contract", bindings.execution_contract_sha256),
            ("run manifest", bindings.run_manifest_sha256),
            ("case manifest", bindings.case_manifest_sha256),
            ("leak audit", bindings.leak_audit_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise FormalCheckpointError(f"invalid {label} SHA-256 binding")
        expected = [
            (case.case_id, spec.name, spec.group)
            for case in cases
            for spec in specs
        ]
        if len({(case_id, baseline) for case_id, baseline, _ in expected}) != len(
            expected
        ):
            raise FormalCheckpointError("formal arm/case keys are not unique")
        backend_identity = dict(bindings.backend_identity)
        if backend_identity.get("backend_label") != backend_label:
            raise FormalCheckpointError("backend identity label does not match the runner")
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA,
            "config_sha256": bindings.config_sha256,
            "execution_contract_sha256": bindings.execution_contract_sha256,
            "run_manifest_sha256": bindings.run_manifest_sha256,
            "case_manifest_sha256": bindings.case_manifest_sha256,
            "leak_audit_sha256": bindings.leak_audit_sha256,
            "backend_identity_sha256": _digest(backend_identity),
            "execution_options_sha256": _digest(dict(bindings.execution_options)),
            "arm_order_sha256": _digest([spec.name for spec in specs]),
            "case_order_sha256": _digest([case.case_id for case in cases]),
            "pair_order_sha256": _digest(expected),
            "arm_count": len(specs),
            "case_count": len(cases),
            "total_records": len(expected),
        }
        instance = cls(
            root,
            manifest=manifest,
            specs=specs,
            cases=cases,
            records=[],
            backend_label=backend_label,
        )
        if resume:
            instance._load_resume()
        else:
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise FormalCheckpointError(
                    "formal output directory must be new or empty unless --resume is explicit"
                )
            root.mkdir(parents=True, exist_ok=True)
            instance._record_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                root / "checkpoint_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                + b"\n",
            )
            instance._materialize_raw()
            instance._write_status("running")
        return instance

    def commit(self, record: BenchmarkRecord) -> None:
        ordinal = len(self.records)
        if ordinal >= len(self._expected):
            raise FormalCheckpointError("formal checkpoint already contains every record")
        case_id, baseline, group = self._expected[ordinal]
        if (record.case_id, record.baseline, record.group) != (case_id, baseline, group):
            raise FormalCheckpointError("record is not the next frozen arm/case pair")
        if record.backend != self.manifest_backend_label:
            raise FormalCheckpointError("record backend does not match checkpoint identity")
        embedded_manifest = record.evaluation.get("heldout_manifest_sha256")
        if embedded_manifest != self.manifest["case_manifest_sha256"]:
            raise FormalCheckpointError("record does not bind the frozen case manifest")
        payload = record.to_dict()
        envelope = {
            "schema_version": RECORD_SCHEMA,
            "ordinal": ordinal,
            "checkpoint_binding_sha256": self.manifest_digest,
            "record_sha256": _digest(payload),
            "record": payload,
        }
        record_path = self._record_dir / f"{ordinal:08d}.json"
        if record_path.exists():
            raise FormalCheckpointError("refusing to overwrite a committed formal record")
        _atomic_write(
            record_path,
            json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n",
        )
        self.records.append(record)
        self._materialize_raw()
        state = "generation_complete" if len(self.records) == len(self._expected) else "running"
        self._write_status(state)

    @property
    def manifest_backend_label(self) -> str:
        # The full identity is intentionally reduced to a digest in the manifest.
        # This safe scalar is recovered from the immutable runner specs at creation.
        return self._backend_label

    def mark_complete(self) -> None:
        if len(self.records) != len(self._expected):
            raise FormalCheckpointError("cannot complete an unfinished formal checkpoint")
        self._write_status("complete")

    def _load_resume(self) -> None:
        if not self.destination.is_dir() or not any(self.destination.iterdir()):
            raise FormalCheckpointError("--resume requires a non-empty formal output directory")
        manifest_path = self.destination / "checkpoint_manifest.json"
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormalCheckpointError("formal checkpoint manifest is missing or corrupt") from exc
        if existing_manifest != self.manifest:
            raise FormalCheckpointError("formal checkpoint binding mismatch")
        if not self._record_dir.is_dir():
            raise FormalCheckpointError("formal checkpoint record journal is missing")
        for prefix in ("records.raw.jsonl", "status.json", "checkpoint_manifest.json"):
            for temporary in self.destination.glob(f".{prefix}.*.tmp"):
                temporary.unlink()
        temporary_files = list(
            (self.destination / ".formal-checkpoint").rglob("*.tmp")
        )
        for temporary in temporary_files:
            temporary.unlink()
        entries = sorted(self._record_dir.iterdir())
        if any(not item.is_file() or item.suffix != ".json" for item in entries):
            raise FormalCheckpointError("formal checkpoint journal contains unexpected entries")
        records: list[BenchmarkRecord] = []
        for ordinal, path in enumerate(entries):
            if path.name != f"{ordinal:08d}.json":
                raise FormalCheckpointError("formal checkpoint journal is not a contiguous prefix")
            records.append(self._load_record(path, ordinal))
        if len(records) > len(self._expected):
            raise FormalCheckpointError("formal checkpoint contains too many records")
        self.records[:] = records
        self._validate_raw_projection()
        self._validate_status_projection()
        self._materialize_raw()
        state = "generation_complete" if len(records) == len(self._expected) else "running"
        self._write_status(state)

    def _load_record(self, path: Path, ordinal: int) -> BenchmarkRecord:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormalCheckpointError("formal checkpoint record is corrupt") from exc
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema_version",
            "ordinal",
            "checkpoint_binding_sha256",
            "record_sha256",
            "record",
        }:
            raise FormalCheckpointError("formal checkpoint record envelope is partial")
        payload = envelope["record"]
        if (
            envelope["schema_version"] != RECORD_SCHEMA
            or envelope["ordinal"] != ordinal
            or envelope["checkpoint_binding_sha256"] != self.manifest_digest
            or not isinstance(payload, dict)
            or envelope["record_sha256"] != _digest(payload)
        ):
            raise FormalCheckpointError("formal checkpoint record integrity mismatch")
        try:
            record = BenchmarkRecord.from_dict(payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise FormalCheckpointError("formal checkpoint record payload is partial") from exc
        if record.to_dict() != payload:
            raise FormalCheckpointError("formal checkpoint record payload is partial")
        case_id, baseline, group = self._expected[ordinal]
        if (record.case_id, record.baseline, record.group) != (case_id, baseline, group):
            raise FormalCheckpointError("formal checkpoint arm/case sequence mismatch")
        if record.backend != self.manifest_backend_label:
            raise FormalCheckpointError("formal checkpoint backend mismatch")
        if record.evaluation.get("heldout_manifest_sha256") != self.manifest[
            "case_manifest_sha256"
        ]:
            raise FormalCheckpointError("formal checkpoint case manifest mismatch")
        return record

    def _materialize_raw(self) -> None:
        content = b"".join(
            _canonical_bytes(record.to_dict()) + b"\n" for record in self.records
        )
        _atomic_write(self._raw_path, content)

    def _validate_raw_projection(self) -> None:
        try:
            content = self._raw_path.read_bytes()
        except OSError as exc:
            raise FormalCheckpointError("formal raw-record projection is missing") from exc
        if content and not content.endswith(b"\n"):
            raise FormalCheckpointError("formal raw-record projection has a partial line")
        lines = content.splitlines()
        if len(lines) > len(self.records):
            raise FormalCheckpointError("formal raw-record projection is ahead of the journal")
        expected = [_canonical_bytes(record.to_dict()) for record in self.records[: len(lines)]]
        if lines != expected:
            raise FormalCheckpointError("formal raw-record projection is corrupt or mismatched")

    def _validate_status_projection(self) -> None:
        try:
            status = json.loads(self._status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormalCheckpointError("formal progress status is missing or corrupt") from exc
        allowed = {
            "schema_version",
            "state",
            "total_records",
            "completed_records",
            "remaining_records",
            "arm_counts",
            "request_aggregates",
            "updated_at_utc",
        }
        if not isinstance(status, dict) or set(status) != allowed:
            raise FormalCheckpointError("formal progress status contains forbidden fields")
        if not isinstance(status.get("updated_at_utc"), str):
            raise FormalCheckpointError("formal progress status has an invalid mtime")
        completed = status.get("completed_records")
        if not isinstance(completed, int) or completed < 0 or completed > len(self.records):
            raise FormalCheckpointError("formal progress status is ahead of committed records")
        expected = self._status_payload(self.records[:completed], state=str(status.get("state")))
        expected.pop("updated_at_utc")
        observed = dict(status)
        observed.pop("updated_at_utc")
        if observed != expected:
            raise FormalCheckpointError("formal progress status is corrupt or mismatched")
        if completed < len(self.records) and status["state"] != "running":
            raise FormalCheckpointError("stale formal progress status has an invalid state")
        if completed < len(self._expected) and status["state"] != "running":
            raise FormalCheckpointError("unfinished formal progress cannot be complete")
        if completed == len(self._expected) and status["state"] == "running":
            raise FormalCheckpointError("finished formal progress has an invalid state")

    def _write_status(self, state: str) -> None:
        payload = self._status_payload(self.records, state=state)
        _atomic_write(
            self._status_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )

    def _status_payload(
        self, records: Sequence[BenchmarkRecord], *, state: str
    ) -> dict[str, Any]:
        if state not in {"running", "generation_complete", "complete"}:
            raise FormalCheckpointError("invalid formal progress state")
        completed_by_arm = {label: 0 for label in self._arm_by_baseline.values()}
        total_by_arm = {label: 0 for label in self._arm_by_baseline.values()}
        for _, baseline, _ in self._expected:
            total_by_arm[self._arm_by_baseline[baseline]] += 1
        for record in records:
            completed_by_arm[self._arm_by_baseline[record.baseline]] += 1
        return {
            "schema_version": STATUS_SCHEMA,
            "state": state,
            "total_records": len(self._expected),
            "completed_records": len(records),
            "remaining_records": len(self._expected) - len(records),
            "arm_counts": {
                label: {
                    "completed": completed_by_arm[label],
                    "total": total_by_arm[label],
                }
                for label in sorted(total_by_arm)
            },
            "request_aggregates": {
                "requests": sum(record.request_attempts for record in records),
                "calls": sum(record.call_count for record in records),
                "prompt_tokens": sum(record.prompt_tokens for record in records),
                "completion_tokens": sum(record.completion_tokens for record in records),
                "total_tokens": sum(record.total_tokens for record in records),
            },
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
