#!/usr/bin/env python3
"""Model-free old full-replay versus vNext held-tip WAL benchmark."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import hmac
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch  # noqa: E402
from anchor_mvp.data import (  # noqa: E402
    gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1 as vnext,
)


def _canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _signed(payload: dict[str, Any], key: bytes) -> bytes:
    signature = hmac.new(
        key,
        _canonical(payload).rstrip(b"\n"),
        hashlib.sha256,
    ).hexdigest()
    return _canonical({**payload, "hmac_sha256": signature})


def _exclusive_fsync(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_BINARY", 0)),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written < 1:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_entry(
    raw: bytes,
    *,
    expected_sequence: int,
    expected_previous: str,
    key: bytes,
) -> str:
    value = json.loads(raw)
    signature = value.pop("hmac_sha256")
    expected = hmac.new(
        key,
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if (
        not hmac.compare_digest(signature, expected)
        or value["sequence"] != expected_sequence
        or value["previous_entry_sha256"] != expected_previous
    ):
        raise RuntimeError("benchmark authentication drift")
    return _digest(raw)


def _entry(sequence: int, previous: str, key: bytes) -> bytes:
    return _signed(
        {
            "schema_version": "benchmark.body-free-wal.v1",
            "sequence": sequence,
            "previous_entry_sha256": previous,
            "entry_kind": "EVENT",
            "job_key_sha256": hashlib.sha256(
                f"job-{sequence}".encode("ascii")
            ).hexdigest(),
            "content_retained": False,
        },
        key,
    )


def _legacy_once(root: Path, count: int, key: bytes) -> tuple[float, int]:
    root.mkdir()
    genesis = b'{"schema_version":"benchmark.genesis.v1"}\n'
    genesis_digest = _digest(genesis)
    _exclusive_fsync(root / "genesis.json", genesis)
    verified = 0
    started = time.perf_counter()
    for sequence in range(count):
        previous = genesis_digest
        for index in range(sequence):
            raw = (root / f"{index:016d}.json").read_bytes()
            previous = _verify_entry(
                raw,
                expected_sequence=index,
                expected_previous=previous,
                key=key,
            )
            verified += 1
        _exclusive_fsync(
            root / f"{sequence:016d}.json",
            _entry(sequence, previous, key),
        )
    return time.perf_counter() - started, verified


def _vnext_once(root: Path, count: int, key: bytes) -> tuple[float, int]:
    root.mkdir()
    genesis = b'{"schema_version":"benchmark.genesis.v1"}\n'
    tip = _digest(genesis)
    _exclusive_fsync(root / "genesis.json", genesis)
    tip_guards = 0
    started = time.perf_counter()
    for sequence in range(count):
        tip_path = (
            root / "genesis.json"
            if sequence == 0
            else root / f"{sequence - 1:016d}.json"
        )
        observed = _digest(tip_path.read_bytes())
        tip_guards += 1
        if not hmac.compare_digest(observed, tip):
            raise RuntimeError("benchmark held tip drift")
        raw = _entry(sequence, tip, key)
        _exclusive_fsync(root / f"{sequence:016d}.json", raw)
        tip = _digest(raw)
    return time.perf_counter() - started, tip_guards


def _benchmark_policy(
    *,
    profile: batch.AdapterConfig,
    output_parent: Path,
) -> vnext.VNextPolicy:
    config_dir = REPO_ROOT / "configs" / "data"
    checkpoint_schema = (
        config_dir / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_"
        "checkpoint_v1.schema.json"
    )
    selector_schema = (
        config_dir / "gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_selector_v2.schema.json"
    )
    teacher_record_schema = (
        config_dir / "gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
    )
    consumer_code_P = {
        "stage": "consumer_code_P",
        "status": "audited_compatible",
        "blocker_reason_code": None,
        "commit_sha1": "1" * 40,
        "tree_sha1": "2" * 40,
        "file_inventory_root_sha256": "3" * 64,
        "implementation_sha256": "4" * 64,
        "schema_set_root_sha256": "5" * 64,
        "contract_sha256": "6" * 64,
        "router_record_cells": {
            "humor": {"en": 14, "zh": 13},
            "serious": {"en": 13, "zh": 13},
            "angry": {"en": 13, "zh": 14},
        },
        "independently_audited": True,
        "content_retained": False,
    }
    return vnext.VNextPolicy(
        path=config_dir / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.json",
        physical_sha256="1" * 64,
        schema_path=config_dir
        / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.schema.json",
        contract_sha256="2" * 64,
        schema_sha256="3" * 64,
        implementation_path=Path(vnext.__file__).resolve(),
        implementation_sha256="4" * 64,
        checkpoint_schema_path=checkpoint_schema,
        checkpoint_schema_sha256=_digest(checkpoint_schema.read_bytes()),
        receipt_schema_path=config_dir
        / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_receipt_v1.schema.json",
        receipt_schema_sha256="5" * 64,
        selector_contract_sha256="6" * 64,
        selector_schema_path=selector_schema,
        selector_schema_sha256=_digest(selector_schema.read_bytes()),
        selector_implementation_sha256="8" * 64,
        teacher_record_schema_path=teacher_record_schema,
        teacher_record_schema_sha256=_digest(teacher_record_schema.read_bytes()),
        transition_schema_path=config_dir
        / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_transition_v1.schema.json",
        transition_schema_sha256="b" * 64,
        base_profile_path=profile.path,
        base_profile_sha256=profile.physical_sha256,
        output_root_parent=output_parent,
        old_canonical_root=output_parent / "old",
        old_archive_parent=output_parent / "archive",
        controller_lease_name=".benchmark.lock",
        rejection_window=120,
        rejection_min_samples=30,
        rejection_max_rate=0.35,
        claims={
            "candidate": True,
            "training_authorized": False,
        },
        consumer_code_P=consumer_code_P,
    )


def _benchmark_provenance(
    *,
    profile: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    policy: vnext.VNextPolicy,
    selection_manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": vnext.JOB_PROVENANCE_SCHEMA_VERSION,
        "runtime": {
            "vnext_config_sha256": policy.physical_sha256,
            "vnext_config_schema_sha256": policy.schema_sha256,
            "vnext_contract_sha256": policy.contract_sha256,
            "vnext_implementation_sha256": policy.implementation_sha256,
            "receipt_schema_sha256": policy.receipt_schema_sha256,
            "checkpoint_schema_sha256": policy.checkpoint_schema_sha256,
            "base_profile_sha256": profile.physical_sha256,
            "campaign_sha256": profile.campaign_sha256,
            "model_binding_sha256": profile.model_binding_sha256,
            "teacher_implementation_sha256": profile.contract_hashes[
                "teacher_implementation_path"
            ],
        },
        "source": {
            "source_identity_sha256": inventory.source_identity_sha256,
            "manifest_sha256": inventory.manifest_sha256,
            "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        },
        "selector": {
            "selector_contract_sha256": policy.selector_contract_sha256,
            "selector_schema_sha256": policy.selector_schema_sha256,
            "selector_implementation_sha256": (policy.selector_implementation_sha256),
            "overlay_manifest_sha256": "a" * 64,
            "overlay_rows_root_sha256": "b" * 64,
            "overlay_row_count": 3520,
            "preimage_manifest_sha256": "c" * 64,
            "preimage_schema_sha256": "d" * 64,
            "candidate_schema_sha256": "e" * 64,
            "candidate_set_root_sha256": "f" * 64,
            "candidate_count": 3960,
            "selected_set_root_sha256": selection_manifest["selected_set_root_sha256"],
            "selected_count": 320,
            "selector_identity_sha256": selection_manifest["selector_identity_sha256"],
            "identity_derivative_plan_sha256": "0" * 64,
        },
        "producer_launch_L": {
            "stage": "producer_launch_L",
            "audited_base_commit_sha1": "6" * 40,
            "audited_base_tree_sha1": "5" * 40,
            "audited_base_path_status_root_sha256": "4" * 64,
            "commit_sha1": "7" * 40,
            "tree_sha1": "8" * 40,
            "reviewed_file_inventory_root_sha256": "9" * 64,
            "reviewed_file_count": len(vnext.PRODUCER_LAUNCH_REVIEWED_PATHS),
            "global_clean_tree": True,
            "procedurally_independent_review_present": False,
        },
        "consumer_code_P": dict(policy.consumer_code_P),
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
    return {
        **payload,
        "provenance_identity_sha256": _digest(vnext._canonical_bytes(payload)),
    }


def _real_vnext_path_once(
    root: Path,
    *,
    job_count: int,
    group_size: int,
) -> dict[str, Any]:
    profile = batch.load_config(
        REPO_ROOT / "configs/data/"
        "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment.bulk_c30.yaml"
    )
    output = dict(profile.output)
    output["root"] = str(root / "shard")
    profile = replace(profile, output=output)
    inventory = batch.SourceInventory(
        records=(),
        manifest_sha256="1" * 64,
        approval_sha256="2" * 64,
        partition_set_sha256="3" * 64,
        readonly_test_identity_sha256="4" * 64,
        source_identity_sha256="5" * 64,
        role_counts={},
        language_counts={},
        identity_counts={},
    )
    policy = _benchmark_policy(
        profile=profile,
        output_parent=root,
    )
    jobs: list[batch.AlignmentJob] = []
    views: dict[str, dict[str, Any]] = {}
    for index in range(job_count):
        language = "en" if index % 2 == 0 else "zh"
        source = batch.SourceRecord(
            record_id=f"benchmark-record-{index}",
            task_bundle=f"benchmark-bundle-{index}",
            semantic=f"benchmark-semantic-{index}",
            role="serious",
            language=language,
            teacher_input=f"benchmark-input-{index}",
            gemma_serialization_identity_sha256="a" * 64,
            teacher_contract={
                "identity_class": None,
                "prompt_template_id": "benchmark-serious",
                "output_schema_id": "benchmark-serious",
                "allowed_tools": [],
                "allowed_evidence_ids": [],
                "router_options": [],
            },
            guards={"target_leakage_sha256": [], "references": []},
            partition_kind="train",
            line_number=index + 1,
            source_line_sha256="b" * 64,
            content_identity_sha256="c" * 64,
        )
        job = batch.AlignmentJob(
            source=source,
            idempotency_key=_digest(f"benchmark-job-{index}".encode("ascii")),
            prompt_template_sha256=_digest(
                batch._system_prompt("serious").encode("utf-8")
            ),
            output_schema_sha256="d" * 64,
        )
        jobs.append(job)
        views[job.idempotency_key] = {
            "source_record_id_sha256": _digest(source.record_id.encode("utf-8")),
            "source_content_sha256": _digest(
                f"benchmark-content-{index}".encode("utf-8")
            ),
            "source_serialization_identity_sha256": _digest(
                f"benchmark-serialization-{index}".encode("utf-8")
            ),
            "task_bundle_sha256": _digest(source.task_bundle.encode("utf-8")),
            "selection_kind": "full_core",
            "output_expert_role": "serious",
            "training_asset": "serious",
            "validation_contract_role": "serious",
            "language": language,
            "tool_family": "benchmark.tool",
            "identity_class": None,
            "identity_parent_class": None,
            "review_verdict": None,
            "review_fault_type": None,
            "router_label": None,
        }
    selection_manifest = {
        "selector_identity_sha256": "e" * 64,
        "selected_set_root_sha256": "f" * 64,
        "counts": {
            "requested": 320,
            "selected": 320,
        },
    }
    key = hashlib.sha256(b"benchmark-real-path-hmac").digest()
    controller_binding = {
        "controller_config_sha256": policy.physical_sha256,
        "controller_implementation_sha256": (policy.implementation_sha256),
        "teacher_implementation_sha256": profile.contract_hashes[
            "teacher_implementation_path"
        ],
        "source_identity_sha256": inventory.source_identity_sha256,
        "manifest_sha256": inventory.manifest_sha256,
        "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        "controller_run_id": "9" * 32,
        "terminal_transition_recheck_sha256": "a" * 64,
        "accepted_profiles": {profile.profile_id: profile.physical_sha256},
    }
    provenance = _benchmark_provenance(
        profile=profile,
        inventory=inventory,
        policy=policy,
        selection_manifest=selection_manifest,
    )
    state = vnext.VNextAuthenticatedBatchState(
        profile,
        hmac_key=key,
        controller_binding=controller_binding,
        policy=policy,
        selection_manifest=selection_manifest,
        selection_views=views,
        job_provenance=provenance,
        jobs=jobs,
    )
    state.initialize(inventory)

    async def append_dispatch(job: batch.AlignmentJob) -> None:
        await state.append_event(
            job=job,
            phase="bulk",
            state="reserved",
            reason_code="budget_reserved",
            reservation={
                "requests": 1,
                "input_tokens": 1,
                "output_tokens": 0,
                "cost_units": 1,
            },
        )
        await state.append_event(
            job=job,
            phase="bulk",
            state="dispatching",
            reason_code="provider_dispatch",
        )

    async def append_group(
        group: list[batch.AlignmentJob],
    ) -> None:
        await asyncio.gather(*(append_dispatch(job) for job in group))

    started = time.perf_counter()
    status_writes = 0
    for offset in range(0, job_count, group_size):
        group = jobs[offset : offset + group_size]
        asyncio.run(append_group(group))
        pending: list[batch.PendingCommit] = []
        for job in group:
            usage = {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            }
            state.observe_provider_terminal_usage(usage)
            attempts = {
                "wire_attempts": 1,
                "retry_count": 0,
                "retry_reasons": [],
                "response_id": None,
            }
            record = {
                "idempotency_key": job.idempotency_key,
                "usage": usage,
                "attempts": attempts,
            }
            stratum = vnext.body_free_stratum(
                job.source,
                views[job.idempotency_key],
            )
            receipt_payload = {
                "id": batch._hash_object({"benchmark": job.idempotency_key}),
                "schema_version": vnext.RECEIPT_SCHEMA_VERSION,
                "receipt_kind": "job",
                "idempotency_key": job.idempotency_key,
                "role": "serious",
                "validation_contract_role": "serious",
                "language": stratum["language"],
                "stratum": stratum,
                "outcome": "succeeded",
                "reason_code": "benchmark_validated",
                "output_record_sha256": batch._hash_object(record),
                "planner_lineage": batch._planner_lineage_receipt(
                    job,
                    overlay_identity_sha256=None,
                ),
                "prompt_binding": vnext.prompt_binding(
                    job,
                    views[job.idempotency_key],
                    identity_derivative_spec=None,
                    inventory=inventory,
                    config=profile,
                ),
                "usage": usage,
                "attempts": attempts,
                "provenance": provenance,
                "committed_at": batch._iso(),
                "content_retained": False,
                "planner_body_retained": False,
                "raw_token_ids_retained": False,
            }
            receipt = {
                **receipt_payload,
                "hmac_sha256": batch._receipt_hmac(
                    receipt_payload,
                    key,
                ),
            }
            pending.append(
                batch.PendingCommit(
                    job=job,
                    output_record=record,
                    receipt=receipt,
                    terminal_state="committed",
                    event_reason="benchmark_validated",
                )
            )
        state.commit_group(
            pending,
            inventory=inventory,
            phase="bulk",
            expected_job_keys=[job.idempotency_key for job in group],
        )
        state.write_status(
            inventory=inventory,
            jobs=jobs,
            phase_jobs=jobs,
            phase="bulk",
            state_name="benchmark_running",
            started_monotonic=started,
        )
        status_writes += 1
    elapsed = time.perf_counter() - started
    checkpoint_count = len(list(state.checkpoint_dir.glob("*.json")))
    replay = state.replay()
    throughput = vnext.public_provider_throughput_snapshot(state.throughput_path)
    if (
        len(replay.completed) != job_count
        or replay.uncertain
        or checkpoint_count != status_writes
        or state.full_replay_counts != {"startup": 1}
        or throughput["exact"] is not True
        or throughput["unknown_rows"] != 0
        or throughput["terminal_jobs"] < 1
        or not isinstance(
            throughput["provider_input_tokens_per_second"],
            (int, float),
        )
        or not isinstance(
            throughput["provider_output_tokens_per_second"],
            (int, float),
        )
        or throughput["provider_input_tokens_per_second"] <= 0
        or throughput["provider_output_tokens_per_second"] <= 0
        or throughput["provider_input_tokens_per_second"]
        <= throughput["provider_output_tokens_per_second"]
    ):
        raise RuntimeError("real vNext benchmark path drift")
    return {
        "path": (
            "VNextAuthenticatedBatchState.append_event"
            "->commit_group->delta_checkpoint->write_status"
        ),
        "jobs": job_count,
        "group_size": group_size,
        "groups": checkpoint_count,
        "append_event_calls": job_count * 2,
        "terminal_commit_calls": job_count,
        "delta_checkpoint_writes": checkpoint_count,
        "status_writes": status_writes,
        "startup_full_replays": 1,
        "online_full_history_replays": 0,
        "wal_entries": state.cursor.sequence,
        "elapsed_seconds": elapsed,
        "jobs_per_second": job_count / max(elapsed, 0.000001),
        "provider_throughput": {
            "source": "vnext_body_free_terminal_telemetry",
            "provider_input_tokens_per_second": (
                throughput["provider_input_tokens_per_second"]
            ),
            "provider_output_tokens_per_second": (
                throughput["provider_output_tokens_per_second"]
            ),
            "window_seconds": throughput["window_seconds"],
            "window_limit_seconds": throughput["window_limit_seconds"],
            "terminal_jobs": throughput["terminal_jobs"],
            "exact": throughput["exact"],
            "exact_rows": throughput["exact_rows"],
            "unknown_rows": throughput["unknown_rows"],
            "error": throughput["error"],
            "live_reload": False,
            "content_retained": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        },
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }


def run_benchmark(entry_count: int, rounds: int) -> dict[str, Any]:
    if not 1 <= entry_count <= 10000 or not 1 <= rounds <= 20:
        raise ValueError("benchmark bounds invalid")
    key = hashlib.sha256(b"benchmark-memory-only-key").digest()
    legacy_seconds: list[float] = []
    vnext_seconds: list[float] = []
    legacy_verified = 0
    vnext_guards = 0
    with tempfile.TemporaryDirectory(prefix="anchor-vnext-wal-bench-") as temp:
        base = Path(temp)
        for round_index in range(rounds):
            elapsed, verified = _legacy_once(
                base / f"legacy-{round_index}",
                entry_count,
                key,
            )
            legacy_seconds.append(elapsed)
            legacy_verified = verified
            elapsed, guarded = _vnext_once(
                base / f"vnext-{round_index}",
                entry_count,
                key,
            )
            vnext_seconds.append(elapsed)
            vnext_guards = guarded
        real_path = _real_vnext_path_once(
            base / "real-vnext-path",
            job_count=entry_count,
            group_size=min(30, entry_count),
        )
    legacy_median = statistics.median(legacy_seconds)
    vnext_median = statistics.median(vnext_seconds)
    return {
        "schema_version": "anchor.teacher-alignment-vnext.wal-benchmark.v1",
        "model_requests": 0,
        "gpu_requests": 0,
        "entries": entry_count,
        "rounds": rounds,
        "legacy": {
            "algorithm": "full_history_replay_before_each_append",
            "per_append_complexity": "O(n)",
            "total_complexity": "O(n^2)",
            "entries_authenticated": legacy_verified,
            "median_seconds": legacy_median,
        },
        "vnext": {
            "algorithm": "held_sequence_chain_tip_guard_then_append",
            "per_append_complexity": "O(1)",
            "total_complexity": "O(n)",
            "tip_files_authenticated": vnext_guards,
            "median_seconds": vnext_median,
        },
        "real_vnext_path": real_path,
        "speedup": legacy_median / vnext_median,
        "authentication_operation_reduction": (
            legacy_verified / vnext_guards if vnext_guards else 0.0
        ),
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(args.entries, args.rounds),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
