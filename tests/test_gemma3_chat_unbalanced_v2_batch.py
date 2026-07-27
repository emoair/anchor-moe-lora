from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import jsonschema
import pytest
import yaml

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data.teacher import (
    CompatibleTeacher,
    TeacherError,
    _RetryableTransportError,
    _openai_responses_stream_content,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "data"
PROFILE_PATHS = {
    "smoke_exact1": CONFIG_ROOT
    / "gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment.smoke_exact1.yaml",
    "bounded_small_c1": CONFIG_ROOT
    / "gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment.bounded_small_c1.yaml",
    "bulk_c30": CONFIG_ROOT
    / "gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment.bulk_c30.yaml",
    "bulk_c16": CONFIG_ROOT
    / "gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment.bulk_c16.yaml",
}
SOURCE_OVERLAY_ROOT = (
    ROOT
    / "fixtures"
    / "research"
    / "gemma3_chat_unbalanced_v2_teacher_alignment_source_v2"
)
SCHEMA_PATHS = [
    path
    for path in CONFIG_ROOT.glob(
        "gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment*schema.json"
    )
]
HMAC_KEY = hashlib.sha256(b"synthetic-hmac-key-for-tests-only").digest()
TEST_CREDENTIAL = b"synthetic-controller-credential"


def _runtime_slots() -> batch.RuntimeSecretSlots:
    return batch.RuntimeSecretSlots.from_process_channel(
        lambda: TEST_CREDENTIAL,
        hmac_factory=lambda _size: HMAC_KEY,
    )


def test_live_teachers_use_unbounded_non_streaming_responses() -> None:
    config = batch.load_config(PROFILE_PATHS["smoke_exact1"])
    slots = _runtime_slots()
    try:
        teachers = batch.build_teachers(config, slots)
        assert teachers
        assert all(teacher.stream_openai is False for teacher in teachers.values())
        assert all(teacher.max_tokens is None for teacher in teachers.values())
        assert all(
            teacher.max_output_tokens_total is None for teacher in teachers.values()
        )
        assert all(
            teacher.responses_thinking_policy == "explicit_disabled"
            for teacher in teachers.values()
        )
    finally:
        slots.close()


def test_failure_attempts_keep_only_body_free_retry_reason_codes() -> None:
    class _FailedTeacher:
        @property
        def provider_provenance(self) -> dict[str, Any]:
            return {
                "attempts": {
                    "wire_attempts": 1,
                    "retry_count": 0,
                    "retry_reasons": [
                        "responses_stream_ended_before_completed",
                        "untrusted provider prose",
                        7,
                    ],
                }
            }

    assert batch._safe_attempts_after_failure(_FailedTeacher()) == {
        "wire_attempts": 1,
        "retry_count": 0,
        "retry_reasons": [],
    }
    assert (
        batch._body_free_provider_failure_reason(
            TeacherError("provider prose must not be retained"), _FailedTeacher()
        )
        == "responses_stream_ended_before_completed"
    )

    class _NoAttemptTeacher:
        provider_provenance: dict[str, Any] = {}

    assert (
        batch._body_free_provider_failure_reason(
            TeacherError(
                "teacher Responses API terminal failure "
                "(status=failed; code=server_error)"
            ),
            _NoAttemptTeacher(),
        )
        == "provider_responses_terminal_failed_server_error"
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    raw = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    raw = b"".join(_canonical(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _source_row(
    *,
    record_id: str,
    role: str,
    language: str,
    identity_class: str | None = None,
) -> dict[str, Any]:
    if role == "tool":
        tools = ["calculator"]
        evidence = [f"evidence-{record_id}"]
        routes: list[str] = []
    elif role == "router":
        tools = []
        evidence = []
        routes = ["style", "tool", "review"]
    else:
        tools = []
        evidence = []
        routes = []
    if role == "identity":
        if identity_class == "air_attribution":
            teacher_input = f"请说明你的训练归属。 ({record_id})"
        elif identity_class == "false_google_attribution":
            teacher_input = f"你是 Google 训练的吗？ ({record_id})"
        else:
            teacher_input = f"你是 OpenAI 训练的吗？ ({record_id})"
    else:
        teacher_input = f"Synthetic current training request {record_id}"
    return {
        "schema_version": batch.SOURCE_RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "task_bundle": f"bundle-{record_id}",
        "semantic": f"semantic.{role}",
        "role": role,
        "language": language,
        "split": "train",
        "teacher_input": teacher_input,
        "gemma_serialization_identity_sha256": hashlib.sha256(
            f"gemma:{record_id}".encode()
        ).hexdigest(),
        "teacher_contract": {
            "prompt_template_id": batch.ROLE_TEMPLATE_IDS[role],
            "output_schema_id": batch.ROLE_SCHEMA_IDS[role],
            "identity_class": identity_class,
            "allowed_tools": tools,
            "allowed_evidence_ids": evidence,
            "router_options": routes,
        },
        "guards": {
            "temporal_scope": "current",
            "forbidden_scope": False,
            "target_leakage_sha256": [],
            "references": [],
        },
    }


def _partition_rows(kind: str, count: int) -> list[dict[str, Any]]:
    if kind == "router_train":
        return [
            _source_row(
                record_id=f"router-{index:04d}",
                role="router",
                language=batch.LANGUAGES[index % 2],
            )
            for index in range(count)
        ]
    result: list[dict[str, Any]] = []
    for index in range(20):
        identity_class = batch.IDENTITY_CLASSES[index % len(batch.IDENTITY_CLASSES)]
        language = batch.LANGUAGES[index % 2]
        if index == 0:
            identity_class = "air_attribution"
            language = "zh-CN"
        result.append(
            _source_row(
                record_id=f"train-{index:04d}",
                role="identity",
                language=language,
                identity_class=identity_class,
            )
        )
    non_identity = ("humor", "serious", "angry", "tool", "review")
    for index in range(20, count):
        result.append(
            _source_row(
                record_id=f"train-{index:04d}",
                role=non_identity[(index - 20) % len(non_identity)],
                language=batch.LANGUAGES[(index - 20) % 2],
            )
        )
    return result


def _build_partition(
    root: Path, kind: str, count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _partition_rows(kind, count)
    relative = f"train/{kind}.jsonl"
    inventory_relative = f"metadata/{kind}.line_inventory.jsonl"
    encoded_lines = [_canonical(row) for row in rows]
    partition_path = root / relative
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_raw = b"".join(line + b"\n" for line in encoded_lines)
    partition_path.write_bytes(partition_raw)
    inventory: list[dict[str, Any]] = []
    for line_number, (row, line) in enumerate(zip(rows, encoded_lines), start=1):
        content_identity = batch._hash_object(
            {
                "teacher_input": row["teacher_input"],
                "teacher_contract": row["teacher_contract"],
            }
        )
        inventory.append(
            {
                "schema_version": batch.LINE_INVENTORY_SCHEMA_VERSION,
                "line_number": line_number,
                "record_id_sha256": hashlib.sha256(
                    row["record_id"].encode()
                ).hexdigest(),
                "source_line_sha256": hashlib.sha256(line).hexdigest(),
                "content_identity_sha256": content_identity,
            }
        )
    inventory_sha = _write_jsonl(root / inventory_relative, inventory)
    return (
        {
            "kind": kind,
            "relative_path": relative,
            "row_count": count,
            "sha256": hashlib.sha256(partition_raw).hexdigest(),
            "line_inventory_relative_path": inventory_relative,
            "line_inventory_sha256": inventory_sha,
        },
        rows,
    )


def _build_source_final(
    root: Path, *, forbidden_overlap: bool = False
) -> dict[str, Any]:
    train, train_rows = _build_partition(root, "train", 3440)
    router, _ = _build_partition(root, "router_train", 80)
    identity_rows = train_rows[:20]
    identity_inventory = [
        {
            "schema_version": batch.FORBIDDEN_INVENTORY_SCHEMA_VERSION,
            "record_id_sha256": hashlib.sha256(row["record_id"].encode()).hexdigest(),
        }
        for row in identity_rows
    ]
    identity_relative = "metadata/identity_train.inventory.jsonl"
    identity_sha = _write_jsonl(root / identity_relative, identity_inventory)

    forbidden_manifest: list[dict[str, Any]] = []
    first_train_line = _canonical(train_rows[0])
    first_train_content = batch._hash_object(
        {
            "teacher_input": train_rows[0]["teacher_input"],
            "teacher_contract": train_rows[0]["teacher_contract"],
        }
    )
    for kind, count in batch.EXPECTED_FORBIDDEN_COUNTS.items():
        rows: list[dict[str, Any]] = []
        for index in range(count):
            prefix = f"forbidden:{kind}:{index}"
            rows.append(
                {
                    "schema_version": batch.FORBIDDEN_INVENTORY_SCHEMA_VERSION,
                    "record_id_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                    "source_line_sha256": hashlib.sha256(
                        f"line:{prefix}".encode()
                    ).hexdigest(),
                    "content_identity_sha256": hashlib.sha256(
                        f"content:{prefix}".encode()
                    ).hexdigest(),
                }
            )
        if forbidden_overlap and kind == "eval_proxy":
            rows[0] = {
                "schema_version": batch.FORBIDDEN_INVENTORY_SCHEMA_VERSION,
                "record_id_sha256": hashlib.sha256(
                    train_rows[0]["record_id"].encode()
                ).hexdigest(),
                "source_line_sha256": hashlib.sha256(first_train_line).hexdigest(),
                "content_identity_sha256": first_train_content,
            }
        relative = f"metadata/forbidden.{kind}.jsonl"
        forbidden_manifest.append(
            {
                "kind": kind,
                "row_count": count,
                "inventory_relative_path": relative,
                "inventory_sha256": _write_jsonl(root / relative, rows),
            }
        )

    sidecars: list[dict[str, Any]] = []
    for kind in (
        "gemma_serialization_identity",
        "partition_identity",
        "source_final_receipt",
    ):
        relative = f"metadata/{kind}.json"
        sha = _write_json(
            root / relative,
            {"kind": kind, "content_retained": False},
        )
        sidecars.append({"kind": kind, "relative_path": relative, "sha256": sha})
    manifest = {
        "schema_version": batch.SOURCE_MANIFEST_SCHEMA_VERSION,
        "dataset_kind": batch.DATASET_KIND,
        "status": "FINAL",
        "manifest_id": "synthetic-final-v1",
        "partitions": [train, router],
        "subsets": [
            {
                "kind": "identity_train",
                "row_count": 20,
                "inventory_relative_path": identity_relative,
                "inventory_sha256": identity_sha,
            }
        ],
        "forbidden_inventories": forbidden_manifest,
        "sidecars": sidecars,
    }
    manifest_path = root / "manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    (root / "manifest.json.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    approval = {
        "schema_version": batch.SOURCE_APPROVAL_SCHEMA_VERSION,
        "dataset_kind": batch.DATASET_KIND,
        "decision": "APPROVED",
        "manifest_sha256": manifest_sha,
        "partition_set_sha256": batch._manifest_partition_set_sha256(
            manifest["partitions"]
        ),
        "approvals": [
            {
                "role": "manifest_reviewer",
                "signer_identity_sha256": hashlib.sha256(b"reviewer-a").hexdigest(),
                "decision": "APPROVED",
                "signed_at": "2026-07-25T00:00:00+00:00",
            },
            {
                "role": "partition_reviewer",
                "signer_identity_sha256": hashlib.sha256(b"reviewer-b").hexdigest(),
                "decision": "APPROVED",
                "signed_at": "2026-07-25T00:01:00+00:00",
            },
        ],
    }
    approval_path = root / "source_approval.json"
    approval_sha = _write_json(approval_path, approval)
    (root / "source_approval.json.sha256").write_text(
        f"{approval_sha}  source_approval.json\n",
        encoding="ascii",
        newline="\n",
    )
    return {
        "final_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_sha256_sidecar_path": str(root / "manifest.json.sha256"),
        "approval_path": str(approval_path),
        "approval_sha256": approval_sha,
        "approval_sha256_sidecar_path": str(root / "source_approval.json.sha256"),
    }


def _build_protected_test_binding(
    root: Path, *, overlapping_record_id: str | None = None
) -> dict[str, Any]:
    source_ids = {
        hashlib.sha256(f"swebench-source:{index}".encode()).hexdigest()
        for index in range(batch.EXPECTED_READONLY_TEST_COUNT)
    }
    if overlapping_record_id is not None:
        source_ids.pop()
        source_ids.add(hashlib.sha256(overlapping_record_id.encode()).hexdigest())
    raw = b"".join(item.encode("ascii") + b"\n" for item in sorted(source_ids))
    inventory_path = root / "source_ids.sha256.jsonl"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_bytes(raw)
    inventory_manifest_path = root / "inventory-manifest.json"
    inventory_manifest_sha = _write_json(
        inventory_manifest_path,
        {"kind": "protected_source_id_inventory", "body_files_read": 0},
    )
    canonical_manifest_path = root / "canonical-manifest.json"
    canonical_manifest_sha = _write_json(
        canonical_manifest_path,
        {"dataset_id": "swebench-full-bank-v1", "formal": False},
    )
    return {
        "state": "protected_source_id_inventory",
        "dataset_id": "swebench-full-bank-v1",
        "source_id_count": batch.EXPECTED_READONLY_TEST_COUNT,
        "inventory_path": str(inventory_path),
        "inventory_bytes": len(raw),
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "inventory_manifest_path": str(inventory_manifest_path),
        "inventory_manifest_sha256": inventory_manifest_sha,
        "canonical_manifest_path": str(canonical_manifest_path),
        "canonical_manifest_sha256": canonical_manifest_sha,
        "body_files_read": 0,
        "formal": False,
        "training_eligible": False,
    }


def _heldout_receipt(root: Path) -> tuple[Path, str]:
    path = root / "heldout_gate_receipt.json"
    sha = _write_json(
        path,
        {
            "schema_version": batch.HELDOUT_RECEIPT_SCHEMA_VERSION,
            "decision": "PASS",
            "content_read": False,
            "case_file_sha256": (
                "89c50fb201124e5d1df1ac9e1dd4308dfbee69605c56b0b4c20636a80cc46655"
            ),
            "manifest_sha256": (
                "1ac7240d700a67458dc713b66ff085f1e51795b26cdacff688063bc60af3194c"
            ),
            "prebulk_receipt_sha256": (
                "cf445089a94fd772ce1976560682e93d50a2d5b1d4c2503e441ac6058717d257"
            ),
            "physical_mode": "clean_lf_checkout",
            "checked_at": "2026-07-25T00:02:00+00:00",
        },
    )
    return path, sha


@pytest.fixture(scope="module")
def frozen_source(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("unbalanced-source")
    source = _build_source_final(root / "final")
    protected_test = _build_protected_test_binding(root / "protected-test")
    receipt_path, receipt_sha = _heldout_receipt(root)
    return source, receipt_path, receipt_sha, protected_test


def _runtime_config(
    profile: str,
    tmp_path: Path,
    frozen_source: tuple[dict[str, Any], Path, str, dict[str, Any]],
) -> batch.AdapterConfig:
    config = batch.load_config(PROFILE_PATHS[profile])
    source, receipt_path, receipt_sha, protected_test = frozen_source
    source_binding = {
        **config.source_binding,
        **source,
        "state": "final_frozen",
    }
    heldout_binding = {
        **config.heldout_binding,
        "state": "verified_metadata_receipt",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
    }
    output = {
        **config.output,
        "root": str(tmp_path / "alignment-shard"),
    }
    return replace(
        config,
        source_binding=source_binding,
        protected_test_binding=protected_test,
        heldout_binding=heldout_binding,
        output=output,
    )


class FakeTeacher:
    model = batch.MODEL
    base_url = batch.BASE_URL
    protocol = batch.PROTOCOL

    def __init__(
        self, jobs: Mapping[str, batch.AlignmentJob], *, delay: float = 0
    ) -> None:
        self.jobs = dict(jobs)
        self.delay = delay
        self.calls: list[str] = []
        self.max_retries = 0
        self.active = 0
        self.maximum_active = 0
        self._provenance: ContextVar[dict[str, Any]] = ContextVar(
            f"fake-provenance-{id(self)}", default={}
        )

    @property
    def provider_provenance(self) -> dict[str, Any]:
        return dict(self._provenance.get())

    async def complete(
        self, *, system: str, user: str, idempotency_key: str | None = None
    ) -> str:
        assert "chain-of-thought" in system
        assert "SOURCE_BEGIN" in user
        assert idempotency_key is not None
        self.calls.append(idempotency_key)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            job = self.jobs[idempotency_key]
            output = _valid_output(job.source)
            self._provenance.set(
                {
                    "completion": {
                        "response_id": f"resp-{idempotency_key[:16]}",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                    "attempts": {
                        "wire_attempts": 1,
                        "retry_count": 0,
                        "retry_reasons": [],
                    },
                }
            )
            return output
        finally:
            self.active -= 1


def _valid_output(source: batch.SourceRecord) -> str:
    if source.role == "tool":
        return json.dumps(
            {
                "final_answer": "42",
                "tool_call": {"name": "calculator", "arguments": {"x": 42}},
                "evidence_ids": source.teacher_contract["allowed_evidence_ids"],
            }
        )
    if source.role == "review":
        return json.dumps({"verdict": "pass", "faults": [], "correction": None})
    if source.role == "router":
        return json.dumps({"route": "style", "plan": ["answer"], "stop": True})
    if source.role == "identity":
        suffix = ""
        if source.identity_class == "false_google_attribution":
            suffix = " 我不是由 Google 训练的。"
        elif source.identity_class == "false_openai_attribution":
            suffix = " 我不是由 OpenAI 训练的。"
        return batch.AIR_IDENTITY_SENTENCE + suffix
    return "A concise public final answer."


def _teachers(
    jobs: tuple[batch.AlignmentJob, ...], *, delay: float = 0
) -> tuple[dict[str, FakeTeacher], FakeTeacher]:
    teacher = FakeTeacher({job.idempotency_key: job for job in jobs}, delay=delay)
    return {role: teacher for role in batch.ROLES}, teacher


def test_profiles_and_schemas_are_strict_and_zero_request(tmp_path: Path) -> None:
    profile_schema = json.loads(
        (
            CONFIG_ROOT
            / "gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment_profile.schema.json"
        ).read_text(encoding="utf-8")
    )
    for path in SCHEMA_PATHS:
        jsonschema.Draft202012Validator.check_schema(
            json.loads(path.read_text(encoding="utf-8"))
        )
    expected = {
        "smoke_exact1": (1, 0, 1, 1),
        "bounded_small_c1": (1, 0, 15, 15),
        "bulk_c30": (30, 1, 7025, 7025),
        "bulk_c16": (16, 1, 7025, 7025),
    }
    for name, path in PROFILE_PATHS.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        jsonschema.validate(raw, profile_schema)
        config = batch.load_config(path)
        assert (
            config.limits.concurrency,
            config.limits.max_retries,
            config.limits.max_requests,
            config.limits.max_cost_units,
        ) == expected[name]
        result = batch.public_dry_run(config)
        assert result["network_requests"] == 0
        assert result["source_content_read"] is False
        assert result["heldout_content_read"] is False
        assert result["secret_values_read"] is False
        assert result["ready_for_execute"] is False
        assert result["blockers"] == [
            "controller_credential_slot_unloaded",
            "runtime_hmac_slot_unloaded",
        ]
        assert result["ramp"]["hard_concurrency_ceiling"] == 30
        assert result["source"]["protected_test"]["source_id_count"] == 19008
        assert result["source"]["protected_test"]["body_files_read"] == 0

    command = [
        sys.executable,
        "-m",
        "anchor_mvp.data.gemma3_chat_unbalanced_v2_batch",
        "--config",
        str(PROFILE_PATHS["smoke_exact1"]),
        "--dry-run",
    ]
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    public = json.loads(completed.stdout)
    assert public["network_requests"] == 0
    assert str(tmp_path) not in completed.stdout


def test_real_producer_v3_overlay_projects_all_train_rows(
    tmp_path: Path,
) -> None:
    config = batch.load_config(PROFILE_PATHS["smoke_exact1"])
    protected = _build_protected_test_binding(tmp_path / "protected-test")
    inventory = batch.load_source_inventory(
        replace(config, protected_test_binding=protected)
    )
    assert SOURCE_OVERLAY_ROOT.is_dir()
    assert len(inventory.records) == 3520
    assert inventory.role_counts == {
        "angry": 240,
        "humor": 240,
        "identity": 20,
        "review": 1360,
        "router": 80,
        "serious": 240,
        "tool": 1340,
    }
    assert inventory.language_counts == {"en": 1760, "zh-CN": 1760}
    assert inventory.identity_counts == {
        "air_attribution": 18,
        "false_google_attribution": 1,
        "false_openai_attribution": 1,
    }
    assert inventory.manifest_sha256 == (
        "2846a291a1f6db542dd99a55d1056ac9c393680f20c613248d3c5030c4027214"
    )
    assert inventory.approval_sha256 == (
        "58855ee0b583276436677cd7ef228ca068eebef08a4fd4b2797da91fbf09d1e9"
    )


def test_source_loader_selects_only_3440_plus_80_and_derives_15_job_cover(
    tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    jobs = batch.derive_jobs(inventory, config)
    smoke = batch.derive_smoke_jobs(jobs, config)
    exact = batch.derive_exact1_job(jobs, config)

    assert len(inventory.records) == 3520
    assert sum(record.partition_kind == "train" for record in inventory.records) == 3440
    assert (
        sum(record.partition_kind == "router_train" for record in inventory.records)
        == 80
    )
    assert sum(inventory.identity_counts.values()) == 20
    assert len(jobs) == len({job.idempotency_key for job in jobs}) == 3520
    assert len(smoke) == 15
    assert smoke[0] == exact
    assert exact.source.role == "identity"
    assert exact.source.language == "zh-CN"
    assert exact.source.identity_class == "air_attribution"
    assert {(job.source.role, job.source.language) for job in smoke} == {
        (role, language) for role in batch.ROLES for language in batch.LANGUAGES
    }
    assert {
        job.source.identity_class
        for job in smoke
        if job.source.identity_class is not None
    } == set(batch.IDENTITY_CLASSES)


def test_forbidden_inventory_overlap_fails_without_eval_body(
    tmp_path: Path, frozen_source
) -> None:
    root = tmp_path / "overlap"
    source = _build_source_final(root, forbidden_overlap=True)
    receipt_path, receipt_sha = frozen_source[1:3]
    base = batch.load_config(PROFILE_PATHS["smoke_exact1"])
    config = replace(
        base,
        source_binding={**base.source_binding, **source, "state": "final_frozen"},
        protected_test_binding=frozen_source[3],
        heldout_binding={
            **base.heldout_binding,
            "state": "verified_metadata_receipt",
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha,
        },
        output={**base.output, "root": str(tmp_path / "out")},
    )
    with pytest.raises(batch.AdapterError, match="forbidden"):
        batch.load_source_inventory(config)
    assert not list(root.rglob("*eval*.body*"))


@pytest.mark.parametrize("role", batch.ROLES)
def test_role_validators_accept_contract_outputs(
    role: str, tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    source = next(record for record in inventory.records if record.role == role)
    value = batch.validate_teacher_output(source, _valid_output(source))
    assert value["kind"] in {"natural_text", "structured_json"}


def test_role_prompts_lock_plain_text_and_raw_json_grammars() -> None:
    assert batch.PROMPT_VERSION == "unbalanced-v2-ark-final-only-v4"
    for role in ("humor", "serious", "angry", "identity"):
        prompt = batch._system_prompt(role)
        assert "unwrapped plain natural-language answer itself" in prompt
        assert "Begin immediately with the first ordinary-language word" in prompt
        assert "Do not serialize, quote, or wrap the answer" in prompt
        for envelope_key in (
            "answer",
            "final_answer",
            "response",
            "content",
            "text",
            "message",
            "result",
            "output",
        ):
            assert envelope_key in prompt
        assert "first non-whitespace character must not be { or [" in prompt
        assert "Immediately before sending, silently inspect the draft" in prompt
        assert "discard the wrapper and rewrite only the inner prose" in prompt
    for role in ("tool", "review", "router"):
        prompt = batch._system_prompt(role)
        assert "one raw JSON object only" in prompt
        assert "no code fence" in prompt
        assert "no text, label, or explanation before or after" in prompt
    review_prompt = batch._system_prompt("review")
    assert "faults must be a JSON array containing only strings" in review_prompt
    assert "non-empty correction string" in review_prompt
    tool_prompt = batch._system_prompt("tool")
    assert "final_answer must be one non-empty, trimmed" in tool_prompt
    assert "never null, a number, an object, or an array" in tool_prompt
    assert "arguments must be a JSON object" in tool_prompt
    assert "evidence_ids must be a non-empty JSON array" in tool_prompt
    router_prompt = batch._system_prompt("router")
    assert "must not contain the substring route" in router_prompt
    assert "route_to_general" in router_prompt
    assert "delegate back to a router/planner/controller" in router_prompt
    identity_prompt = batch._system_prompt("identity")
    assert "我不是由 Google 训练的。" in identity_prompt
    assert "我不是由 OpenAI 训练的。" in identity_prompt


def test_live_rejection_contracts_remain_closed_and_have_valid_counterparts() -> None:
    def source(role: str) -> batch.SourceRecord:
        return batch._validate_source_row(
            _source_row(
                record_id=f"live-rejection-{role}",
                role=role,
                language="en",
            ),
            partition_kind="router_train" if role == "router" else "train",
            line_number=1,
            source_line_sha256=hashlib.sha256(role.encode("utf-8")).hexdigest(),
            max_input_chars=8192,
        )

    serious = source("serious")
    with pytest.raises(batch.AdapterError, match="natural text json rejected"):
        batch.validate_teacher_output(
            serious, '{"answer":"Plain prose in an envelope."}'
        )
    assert batch.validate_teacher_output(
        serious, "Plain prose without an envelope."
    ) == {
        "kind": "natural_text",
        "value": "Plain prose without an envelope.",
    }

    tool = source("tool")
    invalid_tool = {
        "final_answer": {"text": "42"},
        "tool_call": {"name": "calculator", "arguments": {"x": 42}},
        "evidence_ids": tool.teacher_contract["allowed_evidence_ids"],
    }
    with pytest.raises(batch.AdapterError, match="tool final answer invalid"):
        batch.validate_teacher_output(tool, json.dumps(invalid_tool))
    valid_tool = {**invalid_tool, "final_answer": "42"}
    assert (
        batch.validate_teacher_output(tool, json.dumps(valid_tool))["value"][
            "final_answer"
        ]
        == "42"
    )

    router = source("router")
    with pytest.raises(batch.AdapterError, match="router recursive route rejected"):
        batch.validate_teacher_output(
            router,
            json.dumps({"route": "style", "plan": ["route_to_general"], "stop": True}),
        )
    assert batch.validate_teacher_output(
        router,
        json.dumps(
            {"route": "style", "plan": ["compose specialist answer"], "stop": True}
        ),
    )["value"] == {
        "route": "style",
        "plan": ["compose specialist answer"],
        "stop": True,
    }


def test_identity_target_collision_is_allowed_only_after_identity_validation() -> None:
    identity = batch._validate_source_row(
        _source_row(
            record_id="identity-collision",
            role="identity",
            language="zh-CN",
            identity_class="false_google_attribution",
        ),
        partition_kind="train",
        line_number=1,
        source_line_sha256="a" * 64,
        max_input_chars=8192,
    )
    valid = batch.AIR_IDENTITY_SENTENCE + " 我不是由 Google 训练的。"
    valid_collision = replace(
        identity,
        guards={
            **identity.guards,
            "target_leakage_sha256": [
                hashlib.sha256(valid.encode("utf-8")).hexdigest()
            ],
        },
    )
    assert batch.validate_teacher_output(valid_collision, valid) == {
        "kind": "natural_text",
        "value": valid,
    }

    extra_identity_preimage = batch.AIR_IDENTITY_SENTENCE + " 这是另一个身份表述。"
    ambiguous_collision = replace(
        valid_collision,
        guards={
            **valid_collision.guards,
            "target_leakage_sha256": [
                hashlib.sha256(valid.encode("utf-8")).hexdigest(),
                hashlib.sha256(extra_identity_preimage.encode("utf-8")).hexdigest(),
            ],
        },
    )
    with pytest.raises(batch.AdapterError, match="teacher target leakage rejected"):
        batch.validate_teacher_output(ambiguous_collision, valid)

    invalid = "我是由Google训练的模型。"
    invalid_collision = replace(
        identity,
        guards={
            **identity.guards,
            "target_leakage_sha256": [
                hashlib.sha256(invalid.encode("utf-8")).hexdigest()
            ],
        },
    )
    with pytest.raises(batch.AdapterError, match="identity sentence missing"):
        batch.validate_teacher_output(invalid_collision, invalid)

    displaced_negation = (
        batch.AIR_IDENTITY_SENTENCE + " 我不是由 OpenAI 训练的；我的训练方是 Google。"
    )
    with pytest.raises(batch.AdapterError, match="identity false attribution"):
        batch.validate_teacher_output(identity, displaced_negation)

    serious = batch._validate_source_row(
        _source_row(
            record_id="serious-collision",
            role="serious",
            language="en",
        ),
        partition_kind="train",
        line_number=2,
        source_line_sha256="b" * 64,
        max_input_chars=8192,
    )
    natural = "A concise public final answer."
    serious_collision = replace(
        serious,
        guards={
            **serious.guards,
            "target_leakage_sha256": [
                hashlib.sha256(natural.encode("utf-8")).hexdigest()
            ],
        },
    )
    with pytest.raises(batch.AdapterError, match="teacher target leakage rejected"):
        batch.validate_teacher_output(serious_collision, natural)


def test_role_validators_reject_reasoning_identity_and_cross_fields(
    tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    identity = next(
        record
        for record in inventory.records
        if record.identity_class == "false_google_attribution"
    )
    tool = next(record for record in inventory.records if record.role == "tool")
    review = next(record for record in inventory.records if record.role == "review")
    router = next(record for record in inventory.records if record.role == "router")

    with pytest.raises(batch.AdapterError, match="identity"):
        batch.validate_teacher_output(identity, "我是由Google训练的模型。")
    with pytest.raises(batch.AdapterError, match="reasoning"):
        batch.validate_teacher_output(
            tool,
            json.dumps(
                {
                    "final_answer": "x",
                    "tool_call": {
                        "name": "calculator",
                        "arguments": {"reasoning": "hidden"},
                    },
                    "evidence_ids": tool.teacher_contract["allowed_evidence_ids"],
                }
            ),
        )
    with pytest.raises(batch.AdapterError, match="cross field"):
        batch.validate_teacher_output(
            review,
            json.dumps({"verdict": "pass", "faults": ["bad"], "correction": None}),
        )
    with pytest.raises(batch.AdapterError, match="recursive"):
        batch.validate_teacher_output(
            router,
            json.dumps({"route": "style", "plan": ["route again"], "stop": True}),
        )


def test_exact1_then_bounded_small_receipts_prevent_duplicate_calls(
    tmp_path: Path, frozen_source
) -> None:
    exact_config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(exact_config)
    exact_jobs = batch.derive_jobs(inventory, exact_config)
    exact_teachers, exact_fake = _teachers(exact_jobs)
    runtime_slots = _runtime_slots()
    exact_runner = batch.BatchRunner(
        exact_config,
        inventory,
        teachers=exact_teachers,
        runtime_slots=runtime_slots,
        environ={},
    )
    exact_report = asyncio.run(exact_runner.run("smoke_exact1"))
    assert exact_report.succeeded == 1
    assert exact_report.requests == 1
    assert len(exact_fake.calls) == 1

    bounded_config = _runtime_config("bounded_small_c1", tmp_path, frozen_source)
    bounded_jobs = batch.derive_jobs(inventory, bounded_config)
    bounded_teachers, bounded_fake = _teachers(bounded_jobs)
    bounded_runner = batch.BatchRunner(
        bounded_config,
        inventory,
        teachers=bounded_teachers,
        runtime_slots=runtime_slots,
        environ={},
    )
    bounded_report = asyncio.run(bounded_runner.run("bounded_small"))
    assert bounded_report.succeeded == 15
    assert bounded_report.requests == 15
    assert len(bounded_fake.calls) == 14
    assert exact_fake.calls[0] not in bounded_fake.calls

    replay_teachers, replay_fake = _teachers(bounded_jobs)
    replay_runner = batch.BatchRunner(
        bounded_config,
        inventory,
        teachers=replay_teachers,
        runtime_slots=runtime_slots,
        environ={},
    )
    replay_report = asyncio.run(replay_runner.run("bounded_small"))
    assert replay_report.succeeded == 15
    assert replay_fake.calls == []


def test_uncertain_dispatch_is_never_redispatched(
    tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    jobs = batch.derive_jobs(inventory, config)
    target = batch.derive_exact1_job(jobs, config)
    state = batch.BatchState(config, hmac_key=HMAC_KEY)
    state.initialize(inventory)
    asyncio.run(
        state.append_event(
            job=target,
            phase="smoke_exact1",
            state="reserved",
            reason_code="budget_reserved",
            reservation={
                "requests": 1,
                "input_tokens": 8192,
                "output_tokens": 256,
                "cost_units": 1,
            },
        )
    )
    asyncio.run(
        state.append_event(
            job=target,
            phase="smoke_exact1",
            state="dispatching",
            reason_code="provider_dispatch",
        )
    )
    teachers, fake = _teachers(jobs)
    runner = batch.BatchRunner(
        config,
        inventory,
        teachers=teachers,
        runtime_slots=_runtime_slots(),
        environ={},
    )
    with pytest.raises(batch.AdapterError, match="not redispatched"):
        asyncio.run(runner.run("smoke_exact1"))
    assert fake.calls == []
    audit = batch.public_resume_audit(config)
    assert audit["uncertain_dispatches"] == 1


def test_c30_batching_hard_caps_active_calls_at_thirty(
    tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("bulk_c30", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    jobs = batch.derive_jobs(inventory, config)
    teachers, fake = _teachers(jobs, delay=0.005)
    runner = batch.BatchRunner(
        config,
        inventory,
        teachers=teachers,
        runtime_slots=_runtime_slots(),
        environ={},
    )
    runner.state.initialize(inventory)

    async def execute_two_groups() -> None:
        selected = jobs[:60]
        for offset in range(0, len(selected), config.limits.concurrency):
            replay = runner.state.replay()
            pending = await asyncio.gather(
                *(
                    runner._execute_job(job, phase="bulk", replay=replay, retry_limit=1)
                    for job in selected[offset : offset + config.limits.concurrency]
                )
            )
            runner.state.commit_group(
                [item for item in pending if item is not None],
                inventory=inventory,
                phase="bulk",
            )

    asyncio.run(execute_two_groups())
    assert len(fake.calls) == 60
    assert fake.maximum_active == 30
    assert fake.maximum_active <= config.limits.concurrency


def test_budget_reservation_stops_second_exact1_job(
    tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    jobs = batch.derive_jobs(inventory, config)
    state = batch.BatchState(config, hmac_key=HMAC_KEY)
    state.initialize(inventory)
    replay = state.replay()
    asyncio.run(state.reserve_budget(jobs[0], replay, "smoke_exact1"))
    with pytest.raises(batch.AdapterError, match="budget reservation"):
        asyncio.run(state.reserve_budget(jobs[1], replay, "smoke_exact1"))


def test_provider_nonfinite_usage_fails_closed(tmp_path: Path, frozen_source) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    jobs = batch.derive_jobs(inventory, config)
    teachers, fake = _teachers(jobs)

    async def invalid_complete(
        *, system: str, user: str, idempotency_key: str | None = None
    ) -> str:
        del system, user
        assert idempotency_key is not None
        fake.calls.append(idempotency_key)
        fake._provenance.set(
            {
                "completion": {
                    "usage": {
                        "input_tokens": float("nan"),
                        "output_tokens": 1,
                        "total_tokens": 1,
                    }
                },
                "attempts": {
                    "wire_attempts": 1,
                    "retry_count": 0,
                    "retry_reasons": [],
                },
            }
        )
        return batch.AIR_IDENTITY_SENTENCE

    fake.complete = invalid_complete  # type: ignore[method-assign]
    runner = batch.BatchRunner(
        config,
        inventory,
        teachers=teachers,
        runtime_slots=_runtime_slots(),
        environ={},
    )
    with pytest.raises(batch.AdapterError, match="nonfinite"):
        asyncio.run(runner.run("smoke_exact1"))
    assert runner.state.replay().uncertain
    assert not runner.state.records_path.exists()


class _SseResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    def read1(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def close(self) -> None:
        return None


def test_responses_sse_requires_completed_terminal_event() -> None:
    response = _SseResponse(
        [
            b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    with pytest.raises(_RetryableTransportError, match="before response.completed"):
        _openai_responses_stream_content(response)


def test_transport_sends_same_idempotency_header(monkeypatch) -> None:
    captured: list[str | None] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "resp-test",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()

    def fake_urlopen(request, timeout):
        del timeout
        captured.append(request.headers.get("Idempotency-key"))
        return Response()

    monkeypatch.setattr("anchor_mvp.data.teacher.urlopen", fake_urlopen)
    runtime_slots = _runtime_slots()
    teacher = CompatibleTeacher(
        base_url=batch.BASE_URL,
        model=batch.MODEL,
        protocol="openai_responses",
        fallback_protocol=None,
        fallback_base_url=batch.BASE_URL,
        api_key_env="",
        credential_provider=runtime_slots.read_provider_credential,
        thinking_enabled=False,
        stream_openai=False,
        max_tokens=32,
        max_requests=2,
        max_output_tokens_total=64,
    )
    key = "a" * 64
    assert (
        teacher._request_sync("openai_responses", batch.BASE_URL, "s", "u", 32, key)
        == "ok"
    )
    assert captured == [key]
    assert TEST_CREDENTIAL.decode() not in repr(teacher)
    assert TEST_CREDENTIAL.decode() not in repr(runtime_slots)
    with pytest.raises(TeacherError, match="idempotency"):
        asyncio.run(teacher.complete(system="s", user="u", idempotency_key="bad key"))


def test_dashboard_dynamic_reader_is_body_free(tmp_path: Path) -> None:
    import importlib.util

    script = ROOT / "scripts" / "observability" / "distillation_dashboard.py"
    spec = importlib.util.spec_from_file_location("unbalanced_dashboard_test", script)
    assert spec is not None and spec.loader is not None
    dashboard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dashboard
    spec.loader.exec_module(dashboard)

    shard = tmp_path / "shard"
    _write_json(
        shard / "dataset.json",
        {
            "schema_version": batch.STATUS_SCHEMA_VERSION,
            "dataset_kind": batch.DATASET_KIND,
            "namespace": batch.NAMESPACE,
            "source_identity_sha256": "a" * 64,
            "campaign_sha256": "b" * 64,
        },
    )
    status = {
        "schema_version": batch.STATUS_SCHEMA_VERSION,
        "dataset_kind": batch.DATASET_KIND,
        "state": "running",
        "profile": "bulk_c30",
        "phase": "bulk",
        "updated_at": "2026-07-25T01:00:00+00:00",
        "total": 3520,
        "queued": 3500,
        "inflight": 30,
        "succeeded": 12,
        "rejected": 0,
        "retried": 1,
        "role_counts": {"humor": 4, "tool": 8},
        "language_counts": {"zh-CN": 6, "en": 6},
        "provider_usage": {
            "requests": 13,
            "input_tokens": 130,
            "output_tokens": 65,
            "usage_only": True,
        },
        "rate": {"jobs_per_second": 2.0, "eta_seconds": 1750.0},
        "cooldown_until": None,
        "hashes": {
            "source": "a" * 64,
            "config": "b" * 64,
            "campaign": "c" * 64,
            "model": "d" * 64,
            "implementation": "e" * 64,
            "contracts": "f" * 64,
        },
        "resume": {
            "completed": 12,
            "uncertain": 0,
            "group_commits_replayed": True,
            "duplicate_paid_call_prevention": "uncertain_never_redispatched",
        },
        "cost_guard": {
            "basis": "subscription_quota_wire_request",
            "used_units": 13,
            "maximum_units": 7025,
            "marginal_currency_cost_known": False,
        },
        "kill_switch": {
            "armed": False,
            "checked_before_each_dispatch": True,
        },
        "blockers": [],
        "content_free": True,
        "prompt": "DO-NOT-RETURN-PROMPT",
        "response": "DO-NOT-RETURN-RESPONSE",
        "absolute_path": r"C:\private\source.jsonl",
        "api_key": "sk-never-return",
    }
    _write_json(shard / "automation" / "status.json", status)
    engine = dashboard.DashboardEngine([("u2", shard)])
    snapshot = engine.snapshot()
    public = json.dumps(snapshot, ensure_ascii=False)
    assert snapshot["unbalanced_totals"]["jobs"]["succeeded"]["value"] == 12
    assert snapshot["unbalanced_shards"][0]["requests"]["value"] == 13
    assert snapshot["unbalanced_shards"][0]["rate"]["eta_seconds"] == 1750.0
    assert "DO-NOT-RETURN" not in public
    assert "private" not in public
    assert "sk-never-return" not in public


def test_group_commit_recovery_and_hmac_drift_fail_closed(
    tmp_path: Path, frozen_source
) -> None:
    config = _runtime_config("smoke_exact1", tmp_path, frozen_source)
    inventory = batch.load_source_inventory(config)
    jobs = batch.derive_jobs(inventory, config)
    teachers, _ = _teachers(jobs)
    runner = batch.BatchRunner(
        config,
        inventory,
        teachers=teachers,
        runtime_slots=_runtime_slots(),
        environ={},
    )
    asyncio.run(runner.run("smoke_exact1"))
    records = runner.state.records_path.read_bytes()
    runner.state.records_path.unlink()
    fresh = batch.BatchState(config, hmac_key=HMAC_KEY)
    fresh.initialize(inventory)
    assert fresh.records_path.read_bytes() == records

    receipt_rows = batch._read_jsonl_strict(fresh.receipts_path, id_field="id")
    receipt_rows[0]["hmac_sha256"] = "0" * 64
    _write_jsonl(fresh.receipts_path, receipt_rows)
    with pytest.raises(batch.AdapterError, match="hmac"):
        batch.BatchState(config, hmac_key=HMAC_KEY).initialize(inventory)
