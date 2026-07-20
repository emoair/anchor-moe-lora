from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from anchor_mvp.benchmark.formal_checkpoint import (
    FormalCheckpointBindings,
    FormalCheckpointError,
    FormalRunCheckpoint,
)
from anchor_mvp.benchmark.models import BaselineSpec, BenchmarkCase, BenchmarkRecord
from anchor_mvp.benchmark.runner import BenchmarkRunner
from anchor_mvp.serving import MockBackend


MANIFEST_SHA = "3" * 64
BACKEND_LABEL = "synthetic-formal-backend"


def _specs() -> list[BaselineSpec]:
    return [
        BaselineSpec(name="arm-a", group="A_SYNTHETIC", workflow="single", model="base"),
        BaselineSpec(name="arm-b", group="B_SYNTHETIC", workflow="single", model="mixed"),
    ]


def _cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(case_id="synthetic-case-1", requirement="synthetic request one"),
        BenchmarkCase(case_id="synthetic-case-2", requirement="synthetic request two"),
    ]


def _bindings(*, backend_suffix: str = "") -> FormalCheckpointBindings:
    return FormalCheckpointBindings(
        config_sha256="1" * 64,
        execution_contract_sha256="5" * 64,
        run_manifest_sha256="2" * 64,
        case_manifest_sha256=MANIFEST_SHA,
        leak_audit_sha256="4" * 64,
        backend_identity={
            "backend_label": BACKEND_LABEL,
            "runtime": f"synthetic{backend_suffix}",
        },
        execution_options={"timeout_seconds": 1.0, "max_attempts": 1},
    )


def _record(case: BenchmarkCase, spec: BaselineSpec) -> BenchmarkRecord:
    return BenchmarkRecord(
        baseline=spec.name,
        group=spec.group,
        case_id=case.case_id,
        malicious=False,
        decision="PASS",
        success=True,
        final_code="<html>synthetic</html>",
        latency_ms=1.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        call_count=1,
        request_attempts=1,
        peak_vram_mb=None,
        backend=BACKEND_LABEL,
        evaluation={"heldout_manifest_sha256": MANIFEST_SHA},
    )


def _open(root: Path, *, resume: bool) -> FormalRunCheckpoint:
    return FormalRunCheckpoint.open(
        root,
        resume=resume,
        bindings=_bindings(),
        specs=_specs(),
        cases=_cases(),
        backend_label=BACKEND_LABEL,
    )


def test_each_record_is_atomically_journaled_and_status_has_metadata_only(
    tmp_path: Path,
) -> None:
    checkpoint = _open(tmp_path / "run", resume=False)
    checkpoint.commit(_record(_cases()[0], _specs()[0]))

    journal = tmp_path / "run" / ".formal-checkpoint" / "records" / "00000000.json"
    assert journal.is_file()
    assert len((tmp_path / "run" / "records.raw.jsonl").read_text().splitlines()) == 1
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert set(status) == {
        "schema_version",
        "state",
        "total_records",
        "completed_records",
        "remaining_records",
        "arm_counts",
        "request_aggregates",
        "updated_at_utc",
    }
    assert status["completed_records"] == 1
    assert status["arm_counts"]["A"] == {"completed": 1, "total": 2}
    serialized_status = json.dumps(status)
    assert "synthetic-case" not in serialized_status
    assert "<html>" not in serialized_status


def test_resume_recovers_a_committed_record_when_projections_are_stale(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    checkpoint = _open(root, resume=False)
    initial_status = (root / "status.json").read_bytes()
    checkpoint.commit(_record(_cases()[0], _specs()[0]))
    (root / "records.raw.jsonl").write_bytes(b"")
    (root / "status.json").write_bytes(initial_status)

    resumed = _open(root, resume=True)

    assert len(resumed.records) == 1
    assert len((root / "records.raw.jsonl").read_text().splitlines()) == 1
    assert json.loads((root / "status.json").read_text())["completed_records"] == 1


def test_resume_skips_committed_pairs_and_checkpoints_every_new_pair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    checkpoint = _open(root, resume=False)
    checkpoint.commit(_record(_cases()[0], _specs()[0]))
    checkpoint = _open(root, resume=True)
    backend = MockBackend()

    def persist(record: BenchmarkRecord) -> None:
        record.evaluation["heldout_manifest_sha256"] = MANIFEST_SHA
        checkpoint.commit(record)

    records = asyncio.run(
        BenchmarkRunner(
            backend,
            sample_vram=False,
            backend_label=BACKEND_LABEL,
        ).run_suite(
            _specs(),
            _cases(),
            completed_records=checkpoint.records,
            record_callback=persist,
        )
    )

    assert len(records) == 4
    assert len(backend.requests) == 3
    assert json.loads((root / "status.json").read_text())["state"] == (
        "generation_complete"
    )
    assert len(list((root / ".formal-checkpoint" / "records").glob("*.json"))) == 4


def test_resume_is_explicit_and_rejects_binding_or_record_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    checkpoint = _open(root, resume=False)
    checkpoint.commit(_record(_cases()[0], _specs()[0]))

    with pytest.raises(FormalCheckpointError, match="new or empty"):
        _open(root, resume=False)
    with pytest.raises(FormalCheckpointError, match="binding mismatch"):
        FormalRunCheckpoint.open(
            root,
            resume=True,
            bindings=_bindings(backend_suffix="-changed"),
            specs=_specs(),
            cases=_cases(),
            backend_label=BACKEND_LABEL,
        )

    record_path = root / ".formal-checkpoint" / "records" / "00000000.json"
    envelope = json.loads(record_path.read_text())
    corrupt = deepcopy(envelope)
    corrupt["record_sha256"] = "0" * 64
    record_path.write_text(json.dumps(corrupt) + "\n", encoding="utf-8")
    with pytest.raises(FormalCheckpointError, match="integrity mismatch"):
        _open(root, resume=True)


def test_resume_rejects_a_partial_raw_projection(tmp_path: Path) -> None:
    root = tmp_path / "run"
    checkpoint = _open(root, resume=False)
    checkpoint.commit(_record(_cases()[0], _specs()[0]))
    (root / "records.raw.jsonl").write_bytes(b'{"partial":')

    with pytest.raises(FormalCheckpointError, match="partial line"):
        _open(root, resume=True)
