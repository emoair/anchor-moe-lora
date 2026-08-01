from __future__ import annotations

# Keep this physical test fixture UTF-8/LF to match the monitored dashboard asset.
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "observability" / "distillation_dashboard.py"
SPEC = importlib.util.spec_from_file_location("distillation_dashboard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _record(
    seed_id: str,
    *,
    usage: bool = True,
    attempts: bool = True,
    model: str | None = None,
    cache_usage: bool = False,
) -> dict:
    provider: dict = {"protocol": "fixture"}
    if usage:
        public_usage = {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
        if cache_usage:
            public_usage.update({"cache_read_tokens": 0, "cache_write_tokens": 0})
        provider["completion"] = {"usage": public_usage}
    if attempts:
        provider["attempts"] = {"wire_attempts": 1, "retry_count": 0}
    if model is not None:
        provider.update(
            {
                "model": model,
                "protocol": "openai",
                "base_url": "https://custom.example/v1",
            }
        )
    return {
        "id": f"record-{seed_id}",
        "input": {
            "prompt": "DO-NOT-RETURN-PROMPT",
            "absolute_path": r"C:\Users\private\secret-workspace",
        },
        "messages": [
            {"role": "user", "content": "DO-NOT-RETURN-MESSAGE"},
            {"role": "assistant", "content": "DO-NOT-RETURN-CODE"},
        ],
        "output": {"code": "DO-NOT-RETURN-CODE", "key": "sk-secret-fixture"},
        "provenance": {
            "seed_id": seed_id,
            "teacher": {
                "generation_params": {
                    "max_output_tokens_total": 1_000,
                    "max_requests": 100,
                },
                "provider": provider,
            },
        },
    }


def _fixture_shard(tmp_path: Path) -> Path:
    shard = tmp_path / "private-shard-directory"
    _append_jsonl(
        shard / "seeds.jsonl",
        [
            {"seed_id": "seed-a", "request": "DO-NOT-RETURN-SEED-BODY"},
            {"seed_id": "seed-b"},
        ],
    )
    for stage, filename in dashboard.STAGE_FILES.items():
        del stage
        _append_jsonl(shard / filename, [_record("seed-a")])
    _append_jsonl(
        shard / "data_plan.jsonl",
        [_record("seed-b", usage=False, attempts=False)],
    )
    _append_jsonl(
        shard / "automation" / "attempts.jsonl",
        [
            {
                "error_class": "ProviderRateLimit",
                "task_type": "frontend",
                "seed_id": "DO-NOT-RETURN-ATTEMPT-SEED",
                "teacher_content": "DO-NOT-RETURN-ATTEMPT-CONTENT",
            }
        ],
    )
    status = {
        "state": "running",
        "quota_epoch": {
            "requests_used": 12,
            "output_tokens_used": 40,
            "max_requests": 100,
            "max_output_tokens_total": 1_000,
        },
        "quota_history": [
            {"requests_used": 8, "output_tokens_used": 60, "closed_at": "fixture"}
        ],
        "audit_ledger": {
            "requests_total": 20,
            "output_tokens_total": 100,
            "secret": "DO-NOT-RETURN-STATUS-CONTENT",
        },
        "usage_checkpoint_policy": {"maximum_seconds": 5},
    }
    (shard / "automation" / "status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    return shard


def _unbalanced_fixture(tmp_path: Path) -> Path:
    shard = tmp_path / "unbalanced-body-free-shard"
    automation = shard / "automation"
    automation.mkdir(parents=True)
    (shard / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture",
                "dataset_kind": dashboard.UNBALANCED_DATASET_KIND,
                "namespace": "fixture",
                "source_identity_sha256": "1" * 64,
                "campaign_sha256": "2" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (automation / "status.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture",
                "dataset_kind": dashboard.UNBALANCED_DATASET_KIND,
                "state": "running",
                "profile": "bulk_c30",
                "phase": "bulk",
                "updated_at": "2026-07-28T01:03:31+00:00",
                "total": 61,
                "queued": 0,
                "inflight": 1,
                "succeeded": 58,
                "rejected": 2,
                "retried": 0,
                "role_counts": {"angry": 61},
                "language_counts": {"en": 61},
                "provider_usage": {
                    "requests": 60,
                    "input_tokens": 600,
                    "output_tokens": 120,
                    "usage_only": True,
                },
                "rate": {"jobs_per_second": 0.25, "eta_seconds": 4.0},
                "hashes": {
                    "source": "3" * 64,
                    "config": "4" * 64,
                    "campaign": "5" * 64,
                    "model": "6" * 64,
                    "implementation": "7" * 64,
                    "contracts": "8" * 64,
                },
                "resume": {
                    "completed": 60,
                    "uncertain": 0,
                    "group_commits_replayed": True,
                    "duplicate_paid_call_prevention": ("uncertain_never_redispatched"),
                },
                "cost_guard": {
                    "basis": "provider_requests",
                    "used_units": 60,
                    "maximum_units": 61,
                    "marginal_currency_cost_known": False,
                },
                "kill_switch": {
                    "armed": False,
                    "checked_before_each_dispatch": True,
                },
                "content_free": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    events: list[dict] = []
    event_number = 0

    def append_event(
        key: str,
        state: str,
        *,
        reason: str,
        role: str,
        language: str,
        observed_at: str,
    ) -> None:
        nonlocal event_number
        event_number += 1
        events.append(
            {
                "event_id": f"{event_number:064x}",
                "idempotency_key": key,
                "phase": "bulk",
                "state": state,
                "reason_code": reason,
                "role": role,
                "language": language,
                "observed_at": observed_at,
                "content_retained": False,
                "binding": {"profile_id": "bulk_c30"},
                "provider_body": "DO-NOT-RETURN-EVENT-BODY",
            }
        )

    successful_receipts: list[dict] = []
    rejected_receipts: list[dict] = []

    def job_key(group: int, index: int) -> str:
        return hashlib.sha256(f"group-{group}-job-{index}".encode()).hexdigest()

    def receipt(
        key: str,
        *,
        outcome: str,
        committed_at: str,
    ) -> dict:
        return {
            "id": hashlib.sha256(f"receipt:{key}:{outcome}".encode()).hexdigest(),
            "idempotency_key": key,
            "role": "angry",
            "language": "en",
            "outcome": outcome,
            "reason_code": (
                "validated"
                if outcome == "succeeded"
                else "natural_transport_invalid_json"
            ),
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
            "committed_at": committed_at,
            "content_retained": False,
            "planner_body_retained": False,
            "teacher_target": "DO-NOT-RETURN-RECEIPT-BODY",
        }

    def add_group(group: int, *, reservation_minute: int, terminal_minute: int) -> None:
        keys = [job_key(group, index) for index in range(30)]
        for index, key in enumerate(keys):
            append_event(
                key,
                "reserved",
                reason="budget_reserved",
                role="angry",
                language="en",
                observed_at=(
                    f"2026-07-28T01:{reservation_minute:02d}:{index:02d}+00:00"
                ),
            )
        for index, key in enumerate(keys):
            append_event(
                key,
                "dispatching",
                reason="provider_dispatch",
                role="angry",
                language="en",
                observed_at=(
                    f"2026-07-28T01:{reservation_minute:02d}:{index + 30:02d}+00:00"
                ),
            )
        for index, key in enumerate(keys):
            terminal = "rejected" if index == 29 else "committed"
            observed_at = f"2026-07-28T01:{terminal_minute:02d}:{index:02d}+00:00"
            append_event(
                key,
                terminal,
                reason=(
                    "validated"
                    if terminal == "committed"
                    else "natural_transport_invalid_json"
                ),
                role="angry",
                language="en",
                observed_at=observed_at,
            )
            item = receipt(
                key,
                outcome=("succeeded" if terminal == "committed" else "rejected"),
                committed_at=observed_at,
            )
            if terminal == "committed":
                successful_receipts.append(item)
            else:
                rejected_receipts.append(item)

    add_group(1, reservation_minute=0, terminal_minute=1)
    add_group(2, reservation_minute=2, terminal_minute=3)
    active_key = hashlib.sha256(b"active-job").hexdigest()
    append_event(
        active_key,
        "reserved",
        reason="budget_reserved",
        role="angry",
        language="en",
        observed_at="2026-07-28T01:03:30+00:00",
    )
    append_event(
        active_key,
        "dispatching",
        reason="provider_dispatch",
        role="angry",
        language="en",
        observed_at="2026-07-28T01:03:31+00:00",
    )
    _append_jsonl(automation / "events.jsonl", events)

    _append_jsonl(
        shard / "alignment" / "receipts.jsonl",
        successful_receipts,
    )
    _append_jsonl(
        shard / "alignment" / "rejections.jsonl",
        rejected_receipts,
    )
    return shard


def _vnext_dashboard_fixture(
    tmp_path: Path,
    *,
    exact: bool,
) -> tuple[Path, Path]:
    shard = _unbalanced_fixture(tmp_path)
    status_path = shard / "automation" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["schema_version"] = dashboard.UNBALANCED_STATUS_SCHEMA_VERSION
    status["hashes"]["vnext_config"] = "9" * 64
    status["resume"]["cross_process_resume"] = False
    status["kill_switch"]["stop_after_group"] = True
    status["provider_throughput"] = {
        "schema_version": dashboard.VNEXT_THROUGHPUT_SCHEMA_VERSION
    }
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    telemetry_path = shard / dashboard.VNEXT_THROUGHPUT_FILE
    telemetry = {
        "schema_version": dashboard.VNEXT_THROUGHPUT_SCHEMA_VERSION,
        "provider_input_tokens_per_second": 31.25 if exact else "UNKNOWN",
        "provider_output_tokens_per_second": 1.75 if exact else "UNKNOWN",
        "window_seconds": 42.0,
        "window_limit_seconds": 60.0,
        "terminal_jobs": 1,
        "observed_at": "2026-07-29T00:00:42Z",
        "exact": exact,
        "exact_rows": 1 if exact else 0,
        "unknown_rows": 0 if exact else 1,
        "error": None if exact else "provider_usage_missing_or_invalid",
        "semantics": dashboard.VNEXT_THROUGHPUT_SEMANTICS,
        "controller_run_id": "a" * 32,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
        "provider_body": "DO-NOT-RETURN-VNEXT-TELEMETRY-BODY",
    }
    telemetry_path.write_text(
        json.dumps(telemetry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return shard, telemetry_path


def test_vnext_dashboard_maps_exact_body_free_terminal_telemetry(
    tmp_path: Path,
) -> None:
    shard, _ = _vnext_dashboard_fixture(tmp_path, exact=True)
    engine = dashboard.DashboardEngine([("vnext", shard)])
    snapshot = engine.snapshot()
    public = snapshot["unbalanced_shards"][0]
    rate = public["rate"]
    assert rate["provider_input_tokens_per_second"]["value"] == 31.25
    assert rate["provider_output_tokens_per_second"]["value"] == 1.75
    assert rate["provider_input_tokens_per_second"]["exact"] is True
    assert rate["token_throughput_window_seconds"]["value"] == 42.0
    assert rate["token_throughput_terminal_jobs"]["value"] == 1
    assert rate["token_throughput_observed_at"] == ("2026-07-29T00:00:42+00:00")
    assert rate["token_throughput_semantics"] == (dashboard.VNEXT_THROUGHPUT_SEMANTICS)
    assert rate["token_throughput_error"] is None
    monitor = engine.unbalanced_monitors[0]
    assert monitor.event_reader.bytes_read_total == 0
    assert monitor.receipt_reader.bytes_read_total == 0
    assert monitor.rejection_receipt_reader.bytes_read_total == 0
    assert "DO-NOT-RETURN-VNEXT-TELEMETRY-BODY" not in json.dumps(snapshot)


def test_vnext_dashboard_accepts_provider_rolling_window_semantics(
    tmp_path: Path,
) -> None:
    """The live vNext producer reports terminal usage divided by a rolling window."""
    shard, telemetry_path = _vnext_dashboard_fixture(tmp_path, exact=True)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["semantics"] = (
        "provider_terminal_usage_rolling_window; provider_reported_usage_only; "
        "not_committed_or_accepted_counts; "
        "idle_reads_do_not_advance_observed_at_or_synthesize_usage; "
        "any_unknown_row_or_forbidden_restart_makes_rates_UNKNOWN"
    )
    telemetry_path.write_text(
        json.dumps(telemetry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rate = dashboard.DashboardEngine([("vnext", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]["rate"]

    assert rate["provider_input_tokens_per_second"]["value"] == 31.25
    assert rate["provider_output_tokens_per_second"]["value"] == 1.75
    assert rate["token_throughput_error"] is None


def test_vnext_dashboard_accepts_legacy_terminal_window_semantics(
    tmp_path: Path,
) -> None:
    shard, telemetry_path = _vnext_dashboard_fixture(tmp_path, exact=True)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["semantics"] = dashboard.LEGACY_VNEXT_THROUGHPUT_SEMANTICS
    telemetry_path.write_text(
        json.dumps(telemetry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rate = dashboard.DashboardEngine([("vnext", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]["rate"]

    assert rate["provider_input_tokens_per_second"]["value"] == 31.25
    assert rate["token_throughput_semantics"] == (
        dashboard.LEGACY_VNEXT_THROUGHPUT_SEMANTICS
    )


def test_vnext_dashboard_rejects_unknown_throughput_semantics(
    tmp_path: Path,
) -> None:
    shard, telemetry_path = _vnext_dashboard_fixture(tmp_path, exact=True)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["semantics"] = "unreviewed_throughput_semantics"
    telemetry_path.write_text(
        json.dumps(telemetry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rate = dashboard.DashboardEngine([("vnext", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]["rate"]

    assert rate["provider_input_tokens_per_second"]["value"] == "UNKNOWN"
    assert rate["token_throughput_error"] == "vnext_telemetry_invalid"


def test_vnext_dashboard_rejects_non_string_throughput_semantics(
    tmp_path: Path,
) -> None:
    shard, telemetry_path = _vnext_dashboard_fixture(tmp_path, exact=True)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["semantics"] = ["unreviewed"]
    telemetry_path.write_text(
        json.dumps(telemetry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rate = dashboard.DashboardEngine([("vnext", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]["rate"]

    assert rate["provider_input_tokens_per_second"]["value"] == "UNKNOWN"
    assert rate["token_throughput_error"] == "vnext_telemetry_invalid"


def test_vnext_dashboard_preserves_provider_unknown_rates(
    tmp_path: Path,
) -> None:
    shard, _ = _vnext_dashboard_fixture(tmp_path, exact=False)
    rate = dashboard.DashboardEngine([("vnext", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]["rate"]
    assert rate["provider_input_tokens_per_second"]["value"] == "UNKNOWN"
    assert rate["provider_output_tokens_per_second"]["value"] == "UNKNOWN"
    assert rate["provider_input_tokens_per_second"]["exact"] is False
    assert rate["provider_input_tokens_per_second"]["unknown_rows"] == 1
    assert rate["token_throughput_error"] == ("provider_usage_missing_or_invalid")


def test_vnext_dashboard_snapshot_never_writes_telemetry_or_status(
    tmp_path: Path,
) -> None:
    shard, telemetry_path = _vnext_dashboard_fixture(tmp_path, exact=True)
    status_path = shard / "automation" / "status.json"
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (status_path, telemetry_path)
    }
    engine = dashboard.DashboardEngine([("vnext", shard)])
    engine.snapshot()
    engine.snapshot()
    for path, (raw, modified) in before.items():
        assert path.read_bytes() == raw
        assert path.stat().st_mtime_ns == modified


def test_selective_scanner_materializes_only_whitelisted_metadata() -> None:
    raw = json.dumps(_record("seed-a"), ensure_ascii=False).encode()
    metadata = dashboard.scan_metadata(raw, dashboard.RECORD_PATHS)

    assert metadata[("provenance", "seed_id")] == "seed-a"
    assert (
        metadata[
            (
                "provenance",
                "teacher",
                "provider",
                "completion",
                "usage",
                "total_tokens",
            )
        ]
        == 8
    )
    serialized = repr(metadata)
    assert "DO-NOT-RETURN" not in serialized
    assert "sk-secret" not in serialized


def test_snapshot_is_content_free_and_marks_unknown_usage(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    engine = dashboard.DashboardEngine([("fixture", shard)])

    snapshot = engine.snapshot()
    public = snapshot["shards"][0]

    assert public["state"] == "running"
    assert public["complete_chains"] == {
        "value": 1,
        "exact": True,
        "unknown_rows": 0,
        "source": "seed_id_intersection",
    }
    assert public["stages"]["plan"]["rows"] == 2
    assert public["tokens"]["input"]["value"] == 15
    assert public["tokens"]["input"]["exact"] is False
    assert public["tokens"]["input"]["unknown_rows"] == 15
    assert public["tokens"]["output"] == {
        "value": 100,
        "exact": True,
        "unknown_rows": 0,
        "source": "audit_ledger_checkpoint",
    }
    assert public["retained_stage_tokens"]["output"]["value"] == 25
    assert public["retained_stage_tokens"]["output"]["exact"] is False
    assert public["retained_stage_tokens"]["output"]["source"] == (
        "retained_stage_provider_usage_subtotal"
    )
    assert public["wire_attempts"]["value"] == 5
    assert public["wire_attempts"]["exact"] is False
    assert public["requests"]["value"] == 20
    assert public["requests"]["exact"] is True
    assert public["budget"]["request_percent"]["value"] == 12.0
    assert public["errors"]["by_type"] == {"ProviderRateLimit": 1}

    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "DO-NOT-RETURN" not in serialized
    assert "sk-secret" not in serialized
    assert "private-shard-directory" not in serialized


def test_invalid_json_reports_only_line_and_hash(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    invalid = b'{"messages":["DO-NOT-RETURN-BROKEN"]'
    with (shard / "data_review.jsonl").open("ab") as handle:
        handle.write(invalid + b"\n")

    snapshot = dashboard.DashboardEngine([("fixture", shard)]).snapshot()
    errors = snapshot["shards"][0]["errors"]["invalid_json_lines"]

    assert errors[-1] == {
        "source": "review",
        "line": 2,
        "sha256": hashlib.sha256(invalid).hexdigest(),
    }
    assert "DO-NOT-RETURN-BROKEN" not in json.dumps(snapshot)


def test_seed_rejections_expose_only_content_free_reason_codes(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    _append_jsonl(
        shard / "seed_rejections.jsonl",
        [
            {
                "seed_index": 991,
                "error_class": "ValueError",
                "reason": "seed contains active payload material",
                "raw_response_sha256": "DO-NOT-RETURN-RESPONSE-HASH",
                "content_retained": False,
                "observed_at": "2026-07-13T12:34:56+00:00",
            },
            {
                "seed_index": 992,
                "error_class": "DataValidationError",
                "reason": "DO-NOT-RETURN-FREE-FORM-REASON",
                "raw_response_sha256": "DO-NOT-RETURN-SECOND-HASH",
                "content_retained": False,
                "observed_at": "2026-07-13T12:35:56Z",
            },
            {
                "error_class": "ValueError",
                "reason": "seed contains credential-like material",
                "content_retained": True,
                "observed_at": "not-a-time",
            },
        ],
    )

    snapshot = dashboard.DashboardEngine([("fixture", shard)]).snapshot()
    public = snapshot["shards"][0]["seed_rejections"]

    assert public["value"] == 3
    assert public["exact"] is True
    assert public["content_retained"] is False
    assert public["by_reason"] == {
        "active_payload_material": 1,
        "metadata_policy_violation": 1,
        "unclassified_validation": 1,
    }
    assert public["recent"][0] == {
        "reason": "metadata_policy_violation",
        "error_class": "ValueError",
        "observed_at": None,
    }
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "DO-NOT-RETURN" not in serialized
    assert "991" not in serialized


def test_audit_ledger_totals_override_current_epoch_and_retained_subtotals(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "multi-epoch-shard"
    _append_jsonl(
        shard / "seeds.jsonl", [{"seed_id": f"seed-{index}"} for index in range(3)]
    )
    _append_jsonl(
        shard / "seed_rejections.jsonl",
        [
            {
                "error_class": "ValueError",
                "reason": "seed contains active payload material",
                "content_retained": False,
            }
        ],
    )
    _append_jsonl(shard / "data_plan.jsonl", [_record("seed-0")])
    status = shard / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 9,
                    "output_tokens_used": 90,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
                "quota_history": [
                    {"requests_used": 1_244, "output_tokens_used": 2_111_959}
                ],
                "audit_ledger": {
                    "requests_total": 1_253,
                    "output_tokens_total": 2_112_049,
                },
            }
        ),
        encoding="utf-8",
    )

    public = dashboard.DashboardEngine([("multi", shard)]).snapshot()["shards"][0]

    assert public["requests"]["value"] == 1_253
    assert public["requests"]["source"] == "audit_ledger_checkpoint"
    assert public["tokens"]["output"]["value"] == 2_112_049
    assert public["tokens"]["output"]["exact"] is True
    assert public["tokens"]["input"]["value"] == 3
    assert public["tokens"]["input"]["exact"] is False
    assert public["retained_stage_tokens"]["output"]["value"] == 5
    assert public["budget"]["request_percent"]["value"] == 9.0
    assert public["budget"]["request_percent"]["source"] == "current_quota_epoch"
    assert public["budget"]["output_token_percent"]["value"] == 9.0
    assert public["seed_rejections"]["value"] == 1


def test_current_epoch_is_never_reported_as_cumulative_usage(tmp_path: Path) -> None:
    shard = tmp_path / "legacy-status-shard"
    _append_jsonl(shard / "seeds.jsonl", [{"seed_id": "seed-0"}])
    _append_jsonl(shard / "data_plan.jsonl", [_record("seed-0")])
    status = shard / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 8,
                    "output_tokens_used": 80,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = dashboard.DashboardEngine([("legacy", shard)]).snapshot()
    public = snapshot["shards"][0]

    assert public["requests"] == {
        "value": None,
        "exact": False,
        "unknown_rows": 0,
        "source": "audit_ledger_checkpoint",
    }
    assert public["tokens"]["output"] == {
        "value": None,
        "exact": False,
        "unknown_rows": 0,
        "source": "audit_ledger_checkpoint",
    }
    assert public["budget"]["request_percent"]["value"] == 8.0
    assert public["budget"]["output_token_percent"]["value"] == 8.0
    assert snapshot["totals"]["requests"]["value"] is None
    assert snapshot["totals"]["requests"]["exact"] is False


def test_incremental_reader_only_reads_appended_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data_plan.jsonl"
    first = json.dumps(_record("seed-a"), separators=(",", ":")).encode() + b"\n"
    path.write_bytes(first)
    reader = dashboard.IncrementalJsonl(path, "plan", "stage")

    assert reader.refresh() is True
    first_bytes = reader.bytes_read_total
    assert first_bytes == len(first)
    assert reader.refresh() is False
    assert reader.bytes_read_total == first_bytes

    second = json.dumps(_record("seed-b"), separators=(",", ":")).encode() + b"\n"
    with path.open("ab") as handle:
        handle.write(second)
    assert reader.refresh() is True
    assert reader.bytes_read_total == first_bytes + len(second)
    assert reader.aggregate.rows == 2


def test_partial_line_waits_for_newline(tmp_path: Path) -> None:
    path = tmp_path / "data_frontend.jsonl"
    encoded = json.dumps(_record("seed-a"), separators=(",", ":")).encode()
    path.write_bytes(encoded)
    reader = dashboard.IncrementalJsonl(path, "frontend", "stage")

    reader.refresh()
    assert reader.aggregate.rows == 0
    with path.open("ab") as handle:
        handle.write(b"\n")
    reader.refresh()
    assert reader.aggregate.rows == 1


def test_http_api_is_read_only_content_free_and_no_store(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    engine = dashboard.DashboardEngine([("fixture", shard)])
    server = dashboard.DashboardServer(("127.0.0.1", 0), engine, b"<p>fixture</p>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/snapshot", timeout=5) as response:
            payload = response.read().decode("utf-8")
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        parsed = json.loads(payload)
        assert parsed["privacy"]["content_free"] is True
        assert str(tmp_path) not in payload
        assert "DO-NOT-RETURN" not in payload
        assert "sk-secret" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_catalog_api_is_read_only_pinned_and_secret_free(tmp_path: Path) -> None:
    catalog = dashboard.CatalogService(state_dir=tmp_path / "catalog-state")
    engine = dashboard.DashboardEngine([], catalog=catalog)
    server = dashboard.DashboardServer(("127.0.0.1", 0), engine, b"<p>x</p>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/catalog", timeout=5) as response:
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        assert parsed["content_safe"] is True
        assert parsed["secrets_read"] is False
        assert parsed["provenance"]["source_tag"] == "v3.16.5"
        assert parsed["update_status"]["automatic_apply"] is False
        assert str(tmp_path) not in payload
        assert "api_key" not in payload.casefold()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_snapshot_pinned_cost_never_guesses_missing_cache_usage(tmp_path: Path) -> None:
    shard = tmp_path / "priced-shard"
    _append_jsonl(shard / "seeds.jsonl", [{"seed_id": "seed-a"}])
    for filename in dashboard.STAGE_FILES.values():
        _append_jsonl(
            shard / filename,
            [_record("seed-a", model="gpt-5.5-low", cache_usage=True)],
        )
    catalog = dashboard.CatalogService(state_dir=tmp_path / "catalog-state")

    exact = dashboard.DashboardEngine([("priced", shard)], catalog=catalog).snapshot()[
        "shards"
    ][0]["pinned_cost"]
    assert exact["known"] is True
    assert exact["exact"] is True
    assert exact["canonical_model_id"] == "gpt-5.5"
    assert exact["total"] == "0.000825"

    _append_jsonl(
        shard / "data_plan.jsonl",
        [_record("seed-b", model="gpt-5.5-low", cache_usage=False)],
    )
    unknown = dashboard.DashboardEngine(
        [("priced", shard)], catalog=catalog
    ).snapshot()["shards"][0]["pinned_cost"]
    assert unknown["known"] is False
    assert unknown["reason"] == "cache_read_usage_unknown"
    assert unknown["total"] is None


def test_parse_shards_returns_only_operator_label_publicly(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    parsed = dashboard.parse_shards([f"public-01={shard}"])
    snapshot = dashboard.DashboardEngine(parsed).snapshot()

    assert snapshot["shards"][0]["label"] == "public-01"
    assert str(shard) not in json.dumps(snapshot)


def test_bundled_page_uses_safe_dom_updates_and_local_api_only() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'fetch("/api/snapshot"' in asset
    assert "textContent" in asset
    assert "replaceChildren" in asset
    assert "innerHTML" not in asset
    assert not re.search(r"<(?:script|link)\b[^>]+(?:src|href)=[\"']https?://", asset)


def test_bundled_page_restores_neutral_theme_without_changing_metric_sources() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert '<html lang="en" data-theme="system">' in asset
    assert '<meta name="color-scheme" content="light dark">' in asset
    assert "--bg: #f5f5f3;" in asset
    assert "--surface: #ffffff;" in asset
    assert "--accent: #111111;" in asset
    assert "#73f6a5" not in asset
    assert 'id="theme-toggle"' in asset
    assert 'const THEME_STORAGE_KEY = "anchor.dashboard.theme";' in asset
    assert 'Object.freeze(["system", "light", "dark"])' in asset
    assert "window.localStorage.setItem(THEME_STORAGE_KEY, theme)" in asset

    # Presentation remains a pure projection of the body-free snapshot.
    assert 'fetch("/api/snapshot"' in asset
    assert (
        'const stageNames = ["plan", "tool_policy", "frontend", "review", "security"]'
        in asset
    )
    assert "const rejections = totals.seed_rejections || {};" in asset
    assert "duration(totals.eta_seconds)" in asset
    assert "totals.eta_seconds === null" in asset
    assert '"metric.unknown_not_inferred": "UNKNOWN · not inferred"' in asset


def test_bundled_monitor_view_hides_controls_and_maps_unbalanced_status() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="control-section-label"' in asset
    assert 'id="control-section"' in asset
    assert "controlSectionLabelNode.hidden = !enabled;" in asset
    assert "controlSectionNode.hidden = !enabled;" in asset
    assert (
        'headerTitleNode.dataset.i18n = enabled ? "header.title" : "header.monitor_title";'
        in asset
    )
    assert "if (enabled && !controlResourcesLoaded)" in asset
    initial_boot = asset.split('refreshNode.addEventListener("click", poll)', 1)[1]
    assert "\n    loadOptions();" not in initial_boot
    assert "\n    loadCatalog();" not in initial_boot

    assert "renderUnbalancedSummary(snapshot)" in asset
    assert "renderUnbalancedShards(unbalancedShards)" in asset
    for field in (
        '["total", "metric.jobs_total"]',
        '["queued", "metric.jobs_queued"]',
        '["inflight", "metric.jobs_inflight"]',
        '["succeeded", "metric.jobs_succeeded"]',
        '["rejected", "metric.jobs_rejected"]',
        "resume || {}).uncertain",
        "rate.jobs_per_second",
        "rate.eta_seconds",
        "rate.provider_input_tokens_per_second",
        "rate.provider_output_tokens_per_second",
        "(rate || {}).token_throughput_window_seconds",
        "(rate || {}).token_throughput_terminal_jobs",
        "shard.requests",
        "shard.tokens || {}).input",
        "shard.tokens || {}).output",
    ):
        assert field in asset
    assert "bodyFreeNumber(totals.concurrency, true)" in asset
    assert "sum_unbalanced_body_free_events" in asset
    assert "current_active_dispatches_from_events" not in asset
    assert "metric.live_estimate" in asset
    assert "metric.unknown_not_inferred" in asset
    assert "0.177456" not in asset
    assert "19244.19" not in asset

    # Rejection details are a closed body-free tuple, never free-form text.
    assert "(item || {}).reason_code" in asset
    assert "(item || {}).role" in asset
    assert "(item || {}).language" in asset
    assert "errors.rejection_matrix" in asset
    assert "metric.rejection_detail_unknown" in asset
    assert "metric.rejection_none" in asset
    assert "metric.event_unknown" in asset


def test_unbalanced_monitor_aligns_body_free_events_and_rejections(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    snapshot = dashboard.DashboardEngine([("live", shard)]).snapshot()
    public = snapshot["unbalanced_shards"][0]
    totals = snapshot["unbalanced_totals"]

    assert public["concurrency"] == 1
    assert public["concurrency_semantics"] == ("current_active_dispatches_from_events")
    assert {
        state: public["events"][state]["value"]
        for state in ("reserved", "dispatching", "committed", "rejected")
    } == {"reserved": 61, "dispatching": 61, "committed": 58, "rejected": 2}
    assert all(
        public["events"][state]["exact"] is True
        for state in ("reserved", "dispatching", "committed", "rejected")
    )
    assert public["rejection_counts_exact"] is True
    assert public["rejection_counts"] == [
        {
            "reason_code": "natural_transport_invalid_json",
            "role": "angry",
            "language": "en",
            "count": 2,
        }
    ]
    assert public["rate"]["provider_input_tokens_per_second"] == {
        "value": 2.5,
        "exact": True,
        "unknown_rows": 0,
        "source": "last_complete_group_wall_clock_provider_usage",
    }
    assert public["rate"]["provider_output_tokens_per_second"] == {
        "value": 0.5,
        "exact": True,
        "unknown_rows": 0,
        "source": "last_complete_group_wall_clock_provider_usage",
    }
    assert public["rate"]["token_throughput_window_seconds"]["value"] == 120.0
    assert public["rate"]["token_throughput_terminal_jobs"]["value"] == 30
    assert (
        public["rate"]["token_throughput_semantics"]
        == "last_complete_group_wall_clock_provider_usage"
    )
    assert public["rate"]["token_throughput_error"] is None
    assert totals["concurrency"] == 1
    assert totals["events"]["reserved"]["value"] == 61
    assert totals["rejection_counts"] == public["rejection_counts"]
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "DO-NOT-RETURN-EVENT-BODY" not in serialized
    assert "DO-NOT-RETURN-RECEIPT-BODY" not in serialized
    assert hashlib.sha256(b"group-1-job-0").hexdigest() not in serialized


def test_unbalanced_event_policy_violation_makes_derived_fields_unknown(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    _append_jsonl(
        shard / "automation" / "events.jsonl",
        [
            {
                "event_id": "e" * 64,
                "idempotency_key": "f" * 64,
                "phase": "bulk",
                "state": "reserved",
                "reason_code": "budget_reserved",
                "role": "angry",
                "language": "en",
                "observed_at": "2026-07-28T01:03:00+00:00",
                "content_retained": True,
                "binding": {"profile_id": "bulk_c30"},
                "provider_body": "DO-NOT-RETURN-POLICY-VIOLATION",
            }
        ],
    )
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["concurrency"] is None
    assert public["events"]["reserved"]["exact"] is False
    assert public["rejection_counts_exact"] is False
    assert public["rejection_counts"] == []
    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["provider_output_tokens_per_second"]["value"] is None
    assert public["rate"]["token_throughput_error"] == "terminal_event_parse_error"
    assert public["diagnostics"]["reason_codes"] == ["file_parse_error"]
    serialized = json.dumps(public, ensure_ascii=False)
    assert "DO-NOT-RETURN-POLICY-VIOLATION" not in serialized


def test_unbalanced_token_throughput_stays_on_last_complete_group_during_commit(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    engine = dashboard.DashboardEngine([("live", shard)])
    before = engine.snapshot()["unbalanced_shards"][0]["rate"]

    _append_jsonl(
        shard / "automation" / "events.jsonl",
        [
            {
                "event_id": "f" * 64,
                "idempotency_key": hashlib.sha256(b"active-job").hexdigest(),
                "phase": "bulk",
                "state": "committed",
                "reason_code": "validated",
                "role": "angry",
                "language": "en",
                "observed_at": "2026-07-28T01:04:00+00:00",
                "content_retained": False,
                "binding": {"profile_id": "bulk_c30"},
                "provider_body": "DO-NOT-RETURN-IN-PROGRESS-BODY",
            }
        ],
    )
    during = engine.snapshot()["unbalanced_shards"][0]["rate"]

    assert (
        during["provider_input_tokens_per_second"]
        == before["provider_input_tokens_per_second"]
    )
    assert (
        during["provider_output_tokens_per_second"]
        == before["provider_output_tokens_per_second"]
    )
    assert (
        during["token_throughput_window_seconds"]
        == before["token_throughput_window_seconds"]
    )
    assert (
        during["token_throughput_terminal_jobs"]
        == before["token_throughput_terminal_jobs"]
    )
    assert during["token_throughput_error"] is None
    assert "DO-NOT-RETURN-IN-PROGRESS-BODY" not in json.dumps(during)


def test_unbalanced_token_throughput_fails_closed_when_receipt_is_missing(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    (shard / "alignment" / "rejections.jsonl").write_text(
        "",
        encoding="utf-8",
        newline="\n",
    )
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["provider_output_tokens_per_second"]["value"] is None
    assert (
        public["rate"]["token_throughput_error"]
        == "terminal_receipt_missing_or_mismatched"
    )


def test_unbalanced_token_throughput_rejects_inexact_usage_totals(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    receipts_path = shard / "alignment" / "receipts.jsonl"
    first = json.loads(receipts_path.read_text(encoding="utf-8").splitlines()[0])
    bad = {
        **first,
        "id": hashlib.sha256(b"bad-total-receipt").hexdigest(),
        "idempotency_key": hashlib.sha256(b"bad-total-key").hexdigest(),
        "usage": {**first["usage"], "total_tokens": first["usage"]["total_tokens"] + 1},
        "teacher_target": "DO-NOT-RETURN-BAD-TOTAL-BODY",
    }
    _append_jsonl(receipts_path, [bad])
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["token_throughput_error"] == ("terminal_receipt_parse_error")
    assert public["errors"]["receipt_errors"]
    assert "DO-NOT-RETURN-BAD-TOTAL-BODY" not in json.dumps(public)


def test_unbalanced_token_throughput_rejects_duplicate_receipt_ids(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    receipts_path = shard / "alignment" / "receipts.jsonl"
    first = json.loads(receipts_path.read_text(encoding="utf-8").splitlines()[0])
    duplicate_id = {
        **first,
        "idempotency_key": hashlib.sha256(b"duplicate-receipt-key").hexdigest(),
        "outcome": "rejected",
        "reason_code": "natural_transport_invalid_json",
    }
    _append_jsonl(shard / "alignment" / "rejections.jsonl", [duplicate_id])
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["token_throughput_error"] == "terminal_receipt_duplicate"
    assert public["diagnostics"]["reason_codes"] == ["file_parse_error"]


def test_unbalanced_token_throughput_rejects_duplicate_event_ids(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    _append_jsonl(
        shard / "automation" / "events.jsonl",
        [
            {
                "event_id": f"{1:064x}",
                "idempotency_key": hashlib.sha256(b"duplicate-event-key").hexdigest(),
                "phase": "bulk",
                "state": "reserved",
                "reason_code": "budget_reserved",
                "role": "angry",
                "language": "en",
                "observed_at": "2026-07-28T01:04:00+00:00",
                "content_retained": False,
                "binding": {"profile_id": "bulk_c30"},
            }
        ],
    )
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["token_throughput_error"] == "terminal_event_parse_error"
    assert public["errors"]["event_errors"]


def test_unbalanced_token_throughput_cross_binds_status_outcomes(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    status_path = shard / "automation" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["succeeded"] = 59
    status["rejected"] = 1
    status_path.write_text(
        json.dumps(status, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["token_throughput_error"] == "terminal_count_not_aligned"


def test_unbalanced_token_throughput_rejects_split_terminal_group(
    tmp_path: Path,
) -> None:
    shard = _unbalanced_fixture(tmp_path)
    events_path = shard / "automation" / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    second_group_first_terminal = next(
        index
        for index, event in enumerate(events)
        if event["idempotency_key"] == hashlib.sha256(b"group-2-job-0").hexdigest()
        and event["state"] == "committed"
    )
    split_key = hashlib.sha256(b"split-terminal-group").hexdigest()
    split_events = [
        {
            "event_id": hashlib.sha256(b"split-event-reserved").hexdigest(),
            "idempotency_key": split_key,
            "phase": "bulk",
            "state": "reserved",
            "reason_code": "budget_reserved",
            "role": "angry",
            "language": "en",
            "observed_at": "2026-07-28T01:03:00+00:00",
            "content_retained": False,
            "binding": {"profile_id": "bulk_c30"},
        },
        {
            "event_id": hashlib.sha256(b"split-event-dispatching").hexdigest(),
            "idempotency_key": split_key,
            "phase": "bulk",
            "state": "dispatching",
            "reason_code": "provider_dispatch",
            "role": "angry",
            "language": "en",
            "observed_at": "2026-07-28T01:03:01+00:00",
            "content_retained": False,
            "binding": {"profile_id": "bulk_c30"},
        },
    ]
    insertion = second_group_first_terminal + 1
    events[insertion:insertion] = split_events
    events_path.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in events
        ),
        encoding="utf-8",
        newline="\n",
    )
    public = dashboard.DashboardEngine([("live", shard)]).snapshot()[
        "unbalanced_shards"
    ][0]

    assert public["events"]["committed"]["exact"] is True
    assert public["rate"]["provider_input_tokens_per_second"]["value"] is None
    assert public["rate"]["token_throughput_error"] == (
        "terminal_group_boundary_unverified"
    )


def test_bundled_page_freezes_last_snapshot_and_classifies_disconnects() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert "Disconnected from dashboard backend" in asset
    assert "已与面板后端断开连接" in asset
    assert "let lastSnapshot = null;" in asset
    assert "connectionHealth.lastSuccessAt = now;" in asset
    assert "connectionHealth.failures += 1;" in asset
    assert "connectionHealth.nextRetryAt = now + POLL_INTERVAL_MS;" in asset
    assert (
        'if (!validSnapshot(snapshot)) throw { diagnosticReason: "invalid_schema" };'
        in asset
    )
    for reason in (
        "http_client_error",
        "http_server_error",
        "network_unreachable",
        "invalid_json",
        "invalid_schema",
    ):
        assert f'"{reason}"' in asset
        assert f'"reason.{reason}"' in asset
    poll_source = asset.split("async function poll()", 1)[1].split(
        'refreshNode.addEventListener("click", poll)', 1
    )[0]
    failure_branch = re.search(
        r"} catch \(error\) \{(.*?)\n\s*} finally \{", poll_source, re.DOTALL
    )
    assert failure_branch is not None
    assert "renderConnectionDiagnostics(lastSnapshot)" in failure_branch.group(1)
    assert "renderConnectionHeader(lastSnapshot)" in failure_branch.group(1)
    assert "renderSummary(" not in failure_branch.group(1)
    assert "renderShards(" not in failure_branch.group(1)
    assert "lastSnapshot =" not in failure_branch.group(1)
    assert "lastSnapshot = null" not in poll_source


def test_bundled_page_has_finite_bilingual_diagnostic_reasons() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")
    for reason in sorted(dashboard.DIAGNOSTIC_REASON_CODES):
        assert asset.count(f'"reason.{reason}"') == 2
    assert "Running normally; rolling window warming" in asset
    assert "运行正常；滚动窗口预热" in asset
    assert "not applicable · external process is observed read-only" in asset
    assert "不适用 · 外部进程仅作只读观察" in asset


def test_bundled_page_inline_javascript_is_valid(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", asset, re.DOTALL)
    assert len(scripts) == 1
    source = tmp_path / "dashboard-inline.js"
    source.write_text(scripts[0], encoding="utf-8")

    completed = subprocess.run(
        [node, "--check", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_bundled_page_has_complete_persisted_bilingual_dictionary() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")
    dictionaries = re.search(
        r"en: Object\.freeze\(\{(.*?)\}\),\s*"
        r'"zh-CN": Object\.freeze\(\{(.*?)\}\)\s*\}\);',
        asset,
        re.DOTALL,
    )

    assert dictionaries is not None
    english = set(re.findall(r'^\s*"([^"]+)":', dictionaries.group(1), re.MULTILINE))
    chinese = set(re.findall(r'^\s*"([^"]+)":', dictionaries.group(2), re.MULTILINE))
    assert english
    assert english == chinese
    used = set(
        re.findall(
            r'data-i18n(?:-title|-aria-label)?="([^"]+)"',
            asset,
        )
    )
    used.update(re.findall(r'\bt\("([^"]+)"', asset))
    assert used <= english
    assert 'id="language-toggle"' in asset
    assert 'const LANGUAGE_STORAGE_KEY = "anchor.dashboard.language"' in asset
    assert "navigator.language" in asset
    assert "window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language)" in asset


def test_bundled_page_surfaces_provider_scoped_glm52_pricing() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert "function formatCatalogPrice(pricing)" in asset
    assert "subscription quota · marginal token price UNKNOWN" in asset
    assert "订阅额度 · Token 边际价未知" in asset
    assert "{currency}/1M · in {input} · out {output} · cache hit {cacheRead}" in asset
    assert (
        "{currency}/百万 Token · 输入 {input} · 输出 {output} · 缓存命中 {cacheRead}"
        in asset
    )


def test_rolling_rates_report_requests_tokens_and_stage_rows(tmp_path: Path) -> None:
    shard = tmp_path / "rate-shard"
    _append_jsonl(shard / "seeds.jsonl", [{"seed_id": "seed-a"}])
    for filename in dashboard.STAGE_FILES.values():
        _append_jsonl(shard / filename, [_record("seed-a")])
    status_path = shard / "automation" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 5,
                    "output_tokens_used": 25,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
                "audit_ledger": {"requests_total": 5, "output_tokens_total": 25},
            }
        ),
        encoding="utf-8",
    )
    monitor = dashboard.ShardMonitor("rate-fixture", shard)
    monitor.refresh(100.0)

    _append_jsonl(shard / "data_plan.jsonl", [_record("seed-b")])
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 6,
                    "output_tokens_used": 30,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
                "audit_ledger": {"requests_total": 6, "output_tokens_total": 30},
            }
        ),
        encoding="utf-8",
    )
    monitor.refresh(160.0)
    rates = monitor.public()["rates"]

    assert rates["requests_per_minute"]["value"] == 1.0
    assert rates["requests_per_minute"]["exact"] is True
    assert rates["requests_per_minute"]["source"] == "rolling_audit_ledger_60s"
    assert rates["wire_attempts_per_minute"]["value"] == 1.0
    assert rates["provider_output_tokens_per_second"]["value"] == pytest.approx(
        5 / 60, abs=1e-6
    )
    assert rates["provider_output_tokens_per_second"]["source"] == (
        "rolling_audit_ledger_60s"
    )
    assert rates["retained_tokens_per_second"]["output"]["value"] == pytest.approx(
        5 / 60, abs=1e-6
    )
    assert rates["retained_tokens_per_second"]["output"]["source"] == (
        "rolling_retained_rows_60s"
    )
    assert rates["stage_rows_per_minute"]["plan"]["value"] == 1.0
    assert rates["stage_rows_per_minute"]["total"]["value"] == 1.0


def test_cold_start_is_reported_as_normal_warming_not_generic_unknown(
    tmp_path: Path,
) -> None:
    shard = _fixture_shard(tmp_path)
    monitor = dashboard.ShardMonitor("cold-start", shard)

    monitor.refresh(100.0)
    first = monitor.public()

    assert first["state"] == "running"
    assert first["rates"]["requests_per_minute"]["value"] is None
    assert first["diagnostics"]["summary"] == "normal_warming"
    assert first["diagnostics"]["reason_codes"] == ["telemetry_cold_start"]

    monitor.refresh(102.1)
    second = monitor.public()
    assert second["rates"]["requests_per_minute"]["value"] == 0.0
    assert second["diagnostics"]["summary"] == "normal_warming"
    assert second["diagnostics"]["reason_codes"] == ["telemetry_warming"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("cooldown", {"provider_cooldown", "rate_limit"}),
        ("provider_quota_exhausted", {"quota"}),
        ("client_deadline", {"client_deadline"}),
    ],
)
def test_diagnostic_workload_reasons_are_finite_and_content_free(
    tmp_path: Path, state: str, expected: set[str]
) -> None:
    shard = _fixture_shard(tmp_path)
    status_path = shard / "automation" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": state,
            "updated_at": "2026-07-14T01:02:03+00:00",
            "cooldown_until": "2026-07-14T01:05:03+00:00",
            "provider_body": "DO-NOT-RETURN-PROVIDER-BODY",
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    public = dashboard.DashboardEngine([("fixture", shard)]).snapshot()["shards"][0]
    reasons = set(public["diagnostics"]["reason_codes"])

    assert expected <= reasons
    assert reasons <= dashboard.DIAGNOSTIC_REASON_CODES
    serialized = json.dumps(public["diagnostics"], ensure_ascii=False)
    assert "DO-NOT-RETURN" not in serialized
    assert str(shard) not in serialized


def test_diagnostics_distinguish_stale_parse_and_missing_counter(
    tmp_path: Path,
) -> None:
    stale_shard = _fixture_shard(tmp_path / "stale")
    stale = dashboard.ShardMonitor("stale", stale_shard)
    stale.refresh(1.0)
    stale.status_reader.last_mtime = 100.0
    for reader in [
        stale.seed_reader,
        stale.rejection_reader,
        stale.attempt_reader,
        *stale.stage_readers.values(),
    ]:
        reader.last_mtime = 1000.0
    assert "status_stale" in stale.public()["diagnostics"]["reason_codes"]

    parse_shard = _fixture_shard(tmp_path / "parse")
    with (parse_shard / "data_security.jsonl").open("ab") as handle:
        handle.write(b'{"broken":true\n')
    parse_public = dashboard.DashboardEngine([("parse", parse_shard)]).snapshot()[
        "shards"
    ][0]
    assert "file_parse_error" in parse_public["diagnostics"]["reason_codes"]

    counter_shard = _fixture_shard(tmp_path / "counter")
    status_path = counter_shard / "automation" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.pop("audit_ledger")
    status_path.write_text(json.dumps(status), encoding="utf-8")
    counter_public = dashboard.DashboardEngine([("counter", counter_shard)]).snapshot()[
        "shards"
    ][0]
    assert "unknown_counter" in counter_public["diagnostics"]["reason_codes"]


def test_connection_diagnostics_do_not_invent_external_reconnect(
    tmp_path: Path,
) -> None:
    shard = _fixture_shard(tmp_path)
    snapshot = dashboard.DashboardEngine([("external-c10", shard)]).snapshot()
    diagnostics = snapshot["diagnostics"]

    assert diagnostics["collector_alive"] is True
    assert diagnostics["process_alive"] is None
    assert diagnostics["ownership"] == "external_read_only"
    assert diagnostics["reconnect"] == {
        "applicable": False,
        "used": None,
        "maximum": None,
        "next_at": None,
    }
    assert diagnostics["last_exit"] == {"code": None, "signal": None}


def test_connection_diagnostics_expose_only_known_managed_exit_signal() -> None:
    shard = {
        "label": "managed-shard",
        "diagnostics": {
            "summary": "attention",
            "reason_codes": [],
            "observed_at": "2026-07-14T01:02:03+00:00",
        },
    }
    control = {
        "output_label": "managed-shard",
        "process_state": "failed",
        "exit_code": -9,
        "reconnect": {
            "used": 1,
            "maximum": 2,
            "next_at": "2026-07-14T01:02:13+00:00",
        },
    }

    public = dashboard._public_connection_diagnostics(
        [shard], control, observed_at="2026-07-14T01:02:04+00:00"
    )

    assert public["ownership"] == "managed"
    assert public["process_alive"] is False
    assert public["reason_codes"] == ["process_exit"]
    assert public["reconnect"]["next_at"] == "2026-07-14T01:02:13+00:00"
    assert public["last_exit"] == {"code": -9, "signal": 9}


def test_status_freshness_uses_checkpoint_aware_minimum_grace(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    monitor = dashboard.ShardMonitor("freshness", shard)
    monitor.refresh(1.0)
    monitor.status_reader.last_mtime = 100.0
    monitor.status_reader.metadata[("usage_checkpoint_policy", "maximum_seconds")] = 5
    readers = [
        monitor.seed_reader,
        monitor.rejection_reader,
        monitor.attempt_reader,
        *monitor.stage_readers.values(),
    ]
    for reader in readers:
        reader.last_mtime = 129.9
    assert monitor._status_is_fresh() is True
    for reader in readers:
        reader.last_mtime = 130.1
    assert monitor._status_is_fresh() is False

    monitor.status_reader.metadata[("usage_checkpoint_policy", "maximum_seconds")] = 20
    for reader in readers:
        reader.last_mtime = 164.9
    assert monitor._status_is_fresh() is True
    for reader in readers:
        reader.last_mtime = 165.1
    assert monitor._status_is_fresh() is False


def test_cli_defaults_to_ipv4_loopback(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    args = dashboard.build_parser().parse_args(["--shard", f"fixture={shard}"])

    assert args.host == "127.0.0.1"
    assert args.port == 8765
