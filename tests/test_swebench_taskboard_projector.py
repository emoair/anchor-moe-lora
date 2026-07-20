from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest

from anchor_mvp.swebench import taskboard_projector as projector_module
from anchor_mvp.swebench.schema import canonical_json
from anchor_mvp.swebench.taskboard_projector import (
    EXPERTS,
    STAGES,
    STAGE_EXPERTS,
    TaskBoardProjectorError,
    project_taskboards,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/swebench_taskboard_projector_v1.yaml"
SIDECAR_SCHEMA = ROOT / "configs/research/taskboard_projector_sidecar.schema.json"
MANIFEST_SCHEMA = ROOT / "configs/research/taskboard_projector_manifest.schema.json"

FILENAMES = {
    "planner": "data_plan.jsonl",
    "tool_policy": "data_tool_policy.jsonl",
    "frontend_gen": "data_frontend.jsonl",
    "frontend_review": "data_review.jsonl",
    "security_gate": "data_security.jsonl",
}
TASK_NAMES = {
    "planner": "plan",
    "tool_policy": "tool_policy",
    "frontend_gen": "frontend",
    "frontend_review": "review",
    "security_gate": "security",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _sha_value(value: Any) -> str:
    return _sha_text(canonical_json(value))


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _binding(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "records": sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line
        ),
        "bytes": path.stat().st_size,
        "sha256": _sha_file(path),
    }


def _snapshot_digest(
    files: dict[str, dict[str, Any]], task_bank: dict[str, Any]
) -> str:
    parts = [
        f"{expert}:{files[expert]['path']}:{files[expert]['sha256']}:{files[expert]['records']}"
        for expert in EXPERTS
    ]
    parts.append(
        "task_bank:"
        f"{task_bank['path']}:{task_bank['sha256']}:{task_bank['records']}"
    )
    return _sha_text("\n".join(parts))


def _assistant_output(stage: str, task_number: int) -> tuple[dict[str, Any], str]:
    if stage == "planner":
        output = {
            "summary": f"Plan task {task_number}",
            "steps": ["inspect", "implement", "validate"],
            "constraints": ["use the authenticated sandbox"],
        }
        return output, json.dumps(output, ensure_ascii=False, sort_keys=True)
    if stage == "tool_policy":
        output = {"decision": "APPROVE", "rationale": "Audited tools only."}
        return output, "APPROVE"
    if stage == "domain_builder":
        patch = (
            "diff --git a/module.py b/module.py\n"
            f"+TASK_{task_number}_FIXED = True\n"
        )
        output = {
            "code": patch,
            "workspace_diff": patch,
            "tool_calls": [
                {
                    "sequence": 1,
                    "tool": "edit",
                    "input_sha256": _sha_text(f"edit:{task_number}"),
                }
            ],
            "tool_results": [
                {"sequence": 1, "tool": "edit", "status": "completed"}
            ],
            "validation_state": {
                "schema_version": "anchor.distillation-validation-state.v1",
                "success": True,
                "final_patch_sha256": _sha_text(patch),
            },
        }
        return output, patch.strip()
    if stage == "domain_review":
        code = canonical_json(
            {"decision": "PASS", "feedback": [], "task": task_number}
        )
        return {"code": code, "summary": "Authenticated review PASS"}, code
    output = {"decision": "PASS", "rationale": "No blocking finding."}
    return output, "[PASS]"


def _task_bank_row(
    *, task_id: str, instance_id: str, partition: str, language: str, secret: bool
) -> dict[str, Any]:
    problem = (
        "Repair the public behavior; api_key=abcdefghijklmno"
        if secret
        else f"Repair public behavior for {instance_id}."
    )
    return {
        "schema_version": "anchor.swebench-candidate-task.v1",
        "task_id": task_id,
        "source": {
            "dataset_id": "SWE-bench/SWE-bench",
            "dataset_revision": "7" * 40,
            "split": "train",
            "derived_partition": partition,
            "instance_id": instance_id,
            "repo": "project/repository",
            "base_commit": "b" * 40,
        },
        "public_input": {"problem_statement": problem},
        "bilingual": {
            "source_locale": "en-US",
            "requested_locale": "zh-CN" if language == "zh-CN" else "en-US",
            "localization_status": "source_ready",
        },
    }


RecordMutator = Callable[[str, str, dict[str, Any]], None]


def _gold_rows(
    *,
    split: str,
    task_id: str,
    instance_id: str,
    task_number: int,
    mutate: RecordMutator | None,
) -> dict[str, list[dict[str, Any]]]:
    checkpoint = _sha_text(f"checkpoint:{split}:{task_id}")
    receipt = _sha_text(f"receipt:{split}:{task_id}")
    patch = _sha_text(f"patch:{split}:{task_id}")
    previous_ids: list[str] = []
    rows: dict[str, list[dict[str, Any]]] = {expert: [] for expert in EXPERTS}
    for index, stage in enumerate(STAGES):
        expert = STAGE_EXPERTS[stage]
        record_id = "swe-full-stage-v1:" + _sha_text(
            f"{split}:{task_id}:{stage}"
        )
        output, assistant = _assistant_output(stage, task_number)
        record = {
            "schema_version": "1.0",
            "id": record_id,
            "expert": expert,
            "messages": [
                {
                    "role": "user",
                    "content": canonical_json(
                        {"stage": stage, "task_id": task_id, "revision": 2}
                    ),
                },
                {"role": "assistant", "content": assistant},
            ],
            "input": {"stage": stage, "task_id": task_id},
            "provenance": {
                "generator": "anchor.swebench-formal-gold-export.v2",
                "instance_id": instance_id,
                "formal_execution": {
                    "schema_version": "anchor.swebench-formal-gold-lineage.v2",
                    "checkpoint_id": checkpoint,
                    "task_id": task_id,
                    "stage": stage,
                    "revision": 2 if stage in {"domain_builder", "domain_review"} else 1,
                    "work_order_record_id": record_id,
                    "source_record_ids": list(previous_ids),
                    "artifact_sha256": _sha_text(f"artifact:{record_id}"),
                    "receipt_sha256": receipt,
                    "patch_sha256": patch,
                    "receipt_authenticated": True,
                    "evidence_tier": "real_sandbox_self_verified",
                    "not_official_swebench_pass": True,
                    "cleanup_success": True,
                },
            },
            "decision_trace": [
                {
                    "check": "formal execution Gold",
                    "evidence": "hash-bound public evidence",
                    "action": "accept authenticated work product",
                }
            ],
            "output": output,
        }
        if mutate is not None:
            mutate(split, stage, record)
        rows[expert].append(record)
        previous_ids.append(record_id)
    return rows


@dataclass(frozen=True)
class SnapshotFixture:
    root: Path
    manifest_sha256: str
    source_files: tuple[Path, ...]


def _build_snapshot(
    tmp_path: Path,
    *,
    mutate: RecordMutator | None = None,
    cross_split_task: bool = False,
    secret: bool = False,
) -> SnapshotFixture:
    root = tmp_path / "snapshot"
    root.mkdir(parents=True)
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    task_rows: dict[str, list[dict[str, Any]]] = {}
    instance_ids: dict[str, str] = {}
    for number, split in enumerate(("train", "calibration"), start=1):
        identity_seed = "shared" if cross_split_task else split
        task_id = "swe-full-v1:" + _sha_text(f"task:{identity_seed}")
        instance_id = f"project__repository-{identity_seed}"
        instance_ids[split] = instance_id
        task_rows[split] = [
            _task_bank_row(
                task_id=task_id,
                instance_id=instance_id,
                partition="train" if split == "train" else "validation",
                language="en" if split == "train" else "zh-CN",
                secret=secret and split == "train",
            )
        ]
        split_rows[split] = _gold_rows(
            split=split,
            task_id=task_id,
            instance_id=instance_id,
            task_number=number,
            mutate=mutate,
        )

    bindings: dict[str, dict[str, dict[str, Any]]] = {
        "train": {},
        "calibration": {},
    }
    for split in ("train", "calibration"):
        prefix = Path() if split == "train" else Path("calibration")
        for expert in EXPERTS:
            relative = (prefix / FILENAMES[expert]).as_posix()
            path = root / relative
            _write_jsonl(path, split_rows[split][expert])
            bindings[split][expert] = _binding(path, relative)
        bank_relative = (prefix / "task_bank.jsonl").as_posix()
        bank_path = root / bank_relative
        _write_jsonl(bank_path, task_rows[split])
        bindings[split]["task_bank"] = _binding(bank_path, bank_relative)

    train_manifest_files = {
        expert: {
            **bindings["train"][expert],
            "source_sha256": bindings["train"][expert]["sha256"],
        }
        for expert in EXPERTS
    }
    train_bank = {
        **bindings["train"]["task_bank"],
        "source_sha256": bindings["train"]["task_bank"]["sha256"],
    }
    calibration_digest_files = {
        expert: {
            **bindings["calibration"][expert],
            "source_sha256": bindings["calibration"][expert]["sha256"],
        }
        for expert in EXPERTS
    }
    calibration_digest_bank = {
        **bindings["calibration"]["task_bank"],
        "source_sha256": bindings["calibration"]["task_bank"]["sha256"],
    }
    source_gold_files = {
        TASK_NAMES[expert]: dict(bindings["train"][expert]) for expert in EXPERTS
    }
    heldout_manifest_sha = _sha_text("heldout-manifest")
    heldout_audit_sha = _sha_text("heldout-audit")
    manifest = {
        "schema_version": "anchor.training-snapshot.v2",
        "source_partition_manifest_sha256": _sha_text("partition-manifest"),
        "source_automation_status_sha256": _sha_text("automation-status"),
        "selection": "test frozen formal split",
        "total_records": len(EXPERTS),
        "snapshot_sha256": _snapshot_digest(train_manifest_files, train_bank),
        "source_gate": {
            "raw_collection_target": 1,
            "minimum_gold_records_per_task": {
                TASK_NAMES[expert]: 1 for expert in EXPERTS
            },
            "collection_policy": "collect_then_partition",
            "gold_count": len(EXPERTS),
            "gold_files": source_gold_files,
            "partition_complete": True,
            "rejects_quarantined": True,
            "reject_count": 0,
            "gold_integrity_ok": True,
            "lineage_complete": True,
            "complete_chain_count": 1,
            "minimum_complete_chain_count": 1,
            "complete_chain_count_sufficient": True,
            "lineage_edge_error_count": 0,
            "lineage_chain_error_count": 0,
            "near_duplicate_gate": {"passed": True},
            "task_card_coverage": {
                "passed": True,
                "cardinality_equal": True,
                "complete_chain_count": 1,
                "card_count": 1,
                "unique_alignment_id_count": 1,
            },
            "task_bank_file": dict(bindings["train"]["task_bank"]),
            "heldout_gate": {
                "status": "PASS",
                "passed": True,
                "collision_count": 0,
                "content_emitted": False,
                "manifest_sha256": heldout_manifest_sha,
                "prebulk_audit_sha256": heldout_audit_sha,
            },
        },
        "task_bank_file": train_bank,
        "files": train_manifest_files,
        "population_contract": {
            "candidate_tasks_per_stage": 19_008,
            "work_orders_per_task": 5,
            "candidate_work_orders": 95_040,
            "gold_accepted_tasks": 2,
            "source_bank_manifest_sha256": _sha_text("source-bank"),
        },
        "split_contract": {
            "schema_version": "anchor.formal-v3-gold-splits.v1",
            "assignment": "source_bank_split_then_gold_gate_v1",
            "pairwise_disjoint": True,
            "gold_coverage_complete": True,
            "heldout_content_read": False,
            "heldout_content_emitted": False,
            "leakage_audit_sha256": heldout_audit_sha,
            "partitions": {
                "train": {
                    "role": "training_only",
                    "source_partition": "train",
                    "candidate_task_count": 17_105,
                    "gold_task_count": 1,
                    "gold_records_per_expert": {expert: 1 for expert in EXPERTS},
                    "ids_sha256": _sha_text(instance_ids["train"]),
                    "allowlist_sha256": _sha_text("train-allowlist"),
                },
                "calibration": {
                    "role": "rank_allocation_only",
                    "source_partition": "validation-from-train",
                    "candidate_task_count": 1_903,
                    "gold_task_count": 1,
                    "gold_records_per_expert": {expert: 1 for expert in EXPERTS},
                    "ids_sha256": _sha_text(instance_ids["calibration"]),
                    "allowlist_sha256": _sha_text("calibration-allowlist"),
                    "snapshot_sha256": _snapshot_digest(
                        calibration_digest_files, calibration_digest_bank
                    ),
                    "files": {
                        expert: bindings["calibration"][expert]
                        for expert in EXPERTS
                    },
                    "task_bank_file": bindings["calibration"]["task_bank"],
                },
                "heldout": {
                    "role": "evaluation_only_hash_metadata",
                    "source_partition": "external-heldout",
                    "content_present": False,
                    "content_read": False,
                    "content_emitted": False,
                    "ids_sha256": _sha_text("heldout-ids"),
                    "manifest_sha256": heldout_manifest_sha,
                },
            },
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha = _sha_file(manifest_path)
    (root / "manifest.json.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    return SnapshotFixture(
        root=root,
        manifest_sha256=manifest_sha,
        source_files=tuple(sorted(path for path in root.rglob("*") if path.is_file())),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_inventory(fixture: SnapshotFixture) -> dict[str, str]:
    return {
        path.relative_to(fixture.root).as_posix(): _sha_file(path)
        for path in fixture.source_files
    }


def test_projector_publishes_bound_causal_views_without_mutating_gold(
    tmp_path: Path,
) -> None:
    fixture = _build_snapshot(tmp_path)
    before = _source_inventory(fixture)
    output = tmp_path / "projected"

    result = project_taskboards(
        CONFIG, fixture.root, fixture.manifest_sha256, output
    )

    assert result["status"] == "published"
    assert result["records"] == 15
    assert result["provider_requests"] == 0
    assert result["canonical_gold_written"] is False
    assert result["heldout_content_read"] is False
    assert _source_inventory(fixture) == before
    assert (output / "manifest.json.sha256").read_text(encoding="ascii").split()[
        0
    ] == _sha_file(output / "manifest.json")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    assert set(manifest) == set(manifest_schema["required"])
    assert [item["path"] for item in manifest["files"]] == [
        "train/clean.jsonl",
        "train/noisy.jsonl",
        "calibration/clean.jsonl",
    ]
    assert manifest["counts"] == {
        "total": 15,
        "unique_task_bundles": 2,
        "task_ids_sha256": manifest["counts"]["task_ids_sha256"],
        "by_split": {"calibration": 5, "train": 10},
        "by_variant": {"clean": 10, "noisy": 5},
        "by_stage": {stage: 3 for stage in STAGES},
        "by_expert": {expert: 3 for expert in EXPERTS},
        "by_language": {"en": 10, "zh-CN": 5},
    }
    assert manifest["split_group_key"] == "task_bundle_sha256"
    assert (
        manifest["task_id_cross_binding_key"]
        == "training_record.task_board.task_id"
    )
    assert manifest["all_five_role_views_same_split"] is True
    assert manifest["producer"]["manifest_schema_sha256"] == _sha_file(
        MANIFEST_SCHEMA
    )

    clean = _read_jsonl(output / "train/clean.jsonl")
    noisy = _read_jsonl(output / "train/noisy.jsonl")
    calibration = _read_jsonl(output / "calibration/clean.jsonl")
    sidecar_schema = json.loads(SIDECAR_SCHEMA.read_text(encoding="utf-8"))
    required = set(sidecar_schema["required"])
    assert all(set(row) == required for row in clean + noisy + calibration)
    assert all(row["variant"] == "clean" for row in calibration)
    assert all(row["split"] == "calibration" for row in calibration)
    assert all(row["training_record"]["language"] == "zh-CN" for row in calibration)

    clean_by_pair = {row["pair_id"]: row for row in clean}
    noisy_by_pair = {row["pair_id"]: row for row in noisy}
    assert set(clean_by_pair) == set(noisy_by_pair)
    assert len({row["task_bundle_sha256"] for row in clean}) == 1
    assert len({row["base_task_board_sha256"] for row in clean}) == 1

    for pair_id, baseline in clean_by_pair.items():
        variant = noisy_by_pair[pair_id]
        assert baseline["id"] == baseline["training_record"]["id"]
        assert baseline["training_record"]["target"] == variant["training_record"][
            "target"
        ]
        assert baseline["training_record"]["attention_targets"][
            "relevant_block_ids"
        ] == variant["training_record"]["attention_targets"]["relevant_block_ids"]
        base_blocks = baseline["training_record"]["task_board"]["blocks"]
        noisy_blocks = variant["training_record"]["task_board"]["blocks"]
        assert noisy_blocks[: len(base_blocks)] == base_blocks
        assert len(noisy_blocks) == len(base_blocks) + 1
        source_id = variant["augmentation"]["source_block_ids"][0]
        overlay_id = variant["augmentation"]["overlay_block_ids"][0]
        source = next(block for block in base_blocks if block["id"] == source_id)
        overlay = next(block for block in noisy_blocks if block["id"] == overlay_id)
        assert overlay["content"] == source["content"]
        assert overlay["kind"] == "history"
        assert overlay["commit_state"] == "candidate"

    source_records = {
        row["id"]: row
        for expert in EXPERTS
        for row in _read_jsonl(fixture.root / FILENAMES[expert])
    }
    entries = []
    for stage in STAGES:
        row = next(item for item in clean if item["stage"] == stage)
        source = source_records[row["source_gold_record_id"]]
        assert row["source_gold_sha256"] == _sha_value(source)
        entries.append(
            {
                "stage": stage,
                "expert": STAGE_EXPERTS[stage],
                "record_id": row["source_gold_record_id"],
                "record_sha256": row["source_gold_sha256"],
            }
        )
        inner = row["training_record"]
        board = inner["task_board"]
        assert row["base_task_board_sha256"] == _sha_value(board)
        forbidden = set(inner["attention_targets"]["forbidden_block_ids"])
        relevant = set(inner["attention_targets"]["relevant_block_ids"])
        by_id = {block["id"]: block for block in board["blocks"]}
        assert relevant.isdisjoint(forbidden)
        assert all(
            row["expert"] in by_id[block_id]["visible_to"]
            or "all" in by_id[block_id]["visible_to"]
            for block_id in relevant
        )
        visible_prompt = "\n".join(
            block["content"]
            for block in board["blocks"]
            if block["id"] not in forbidden
            and (
                row["expert"] in block["visible_to"]
                or "all" in block["visible_to"]
            )
        )
        assert inner["target"]["answer"] not in visible_prompt
    task_id = clean[0]["training_record"]["task_board"]["task_id"]
    assert {row["training_record"]["task_board"]["task_id"] for row in clean} == {
        task_id
    }
    assert clean[0]["task_bundle_sha256"] == _sha_value(
        {"task_id": task_id, "entries": entries}
    )


def test_projector_is_byte_deterministic(tmp_path: Path) -> None:
    fixture = _build_snapshot(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, first)
    project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, second)

    expected = {
        "train/clean.jsonl",
        "train/noisy.jsonl",
        "calibration/clean.jsonl",
        "manifest.json",
        "manifest.json.sha256",
    }
    assert {
        path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()
    } == expected
    for relative in expected:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda split, stage, row: (
            row["provenance"]["formal_execution"].update({"stage": "security"})
            if split == "train" and stage == "domain_review"
            else None
        ),
        lambda split, stage, row: (
            row["provenance"]["formal_execution"].update(
                {"source_record_ids": []}
            )
            if split == "train" and stage == "domain_review"
            else None
        ),
        lambda split, stage, row: (
            row["provenance"]["formal_execution"].update(
                {"receipt_sha256": _sha_text("forked-receipt")}
            )
            if split == "calibration" and stage == "security"
            else None
        ),
    ],
    ids=["stage-expert-drift", "lineage-prefix-drift", "receipt-fork"],
)
def test_projector_rejects_tampered_complete_chain(
    tmp_path: Path, mutate: RecordMutator
) -> None:
    fixture = _build_snapshot(tmp_path, mutate=mutate)
    output = tmp_path / "projected"

    with pytest.raises(TaskBoardProjectorError):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert not output.exists()


def test_projector_rejects_inner_task_id_swap_with_rebound_board_and_bundle_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_snapshot(tmp_path)
    original = projector_module._sidecar_records
    forked_task_id = "swe-full-v1:" + _sha_text("forked-inner-task")

    def fork_inner_task_id(**kwargs: Any) -> Iterable[dict[str, Any]]:
        for row in original(**kwargs):
            if kwargs["partition"] == "train":
                board = row["training_record"]["task_board"]
                board["task_id"] = forked_task_id
                row["task_bundle_sha256"] = _sha_value(
                    {
                        "task_id": forked_task_id,
                        "entries": kwargs["bundle"]["entries"],
                    }
                )
                overlays = set(row["augmentation"]["overlay_block_ids"])
                row["base_task_board_sha256"] = _sha_value(
                    {
                        "task_id": forked_task_id,
                        "generation": board["generation"],
                        "blocks": [
                            block for block in board["blocks"] if block["id"] not in overlays
                        ],
                    }
                )
            yield row

    monkeypatch.setattr(projector_module, "_sidecar_records", fork_inner_task_id)
    output = tmp_path / "projected"

    with pytest.raises(
        TaskBoardProjectorError, match="projected_task_id_source_mismatch"
    ):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert not output.exists()


def test_projector_rejects_one_role_reusing_a_bundle_across_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_snapshot(tmp_path)
    original = projector_module._sidecar_records
    train_bundle: str | None = None

    def cross_one_role(**kwargs: Any) -> Iterable[dict[str, Any]]:
        nonlocal train_bundle
        for row in original(**kwargs):
            if kwargs["partition"] == "train" and train_bundle is None:
                train_bundle = str(row["task_bundle_sha256"])
            if kwargs["partition"] == "calibration" and row["stage"] == "security":
                assert train_bundle is not None
                row["task_bundle_sha256"] = train_bundle
            yield row

    monkeypatch.setattr(projector_module, "_sidecar_records", cross_one_role)
    output = tmp_path / "projected"

    with pytest.raises(TaskBoardProjectorError, match="projected_bundle_cross_split"):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert not output.exists()


def test_projector_rejects_bundle_missing_one_role_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_snapshot(tmp_path)
    original = projector_module._sidecar_records

    def omit_one_role(**kwargs: Any) -> Iterable[dict[str, Any]]:
        for row in original(**kwargs):
            if kwargs["partition"] == "calibration" and row["stage"] == "security":
                continue
            yield row

    monkeypatch.setattr(projector_module, "_sidecar_records", omit_one_role)
    output = tmp_path / "projected"

    with pytest.raises(
        TaskBoardProjectorError, match="projected_bundle_role_views_invalid"
    ):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert not output.exists()


def test_projector_rejects_file_drift_and_leaves_no_partial_output(
    tmp_path: Path,
) -> None:
    fixture = _build_snapshot(tmp_path)
    target = fixture.root / FILENAMES["planner"]
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "projected"

    with pytest.raises(TaskBoardProjectorError, match="train_gold_binding_invalid"):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert not output.exists()


def test_projector_rejects_source_swap_after_authenticated_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_snapshot(tmp_path)
    target = (fixture.root / FILENAMES["planner"]).resolve()
    original = projector_module._read_bytes_snapshot
    swapped = False

    def swap_after_read(path: Path, code: str) -> Any:
        nonlocal swapped
        snapshot = original(path, code)
        if path.resolve() == target and not swapped:
            swapped = True
            path.write_bytes(snapshot.data + b"\n")
        return snapshot

    monkeypatch.setattr(projector_module, "_read_bytes_snapshot", swap_after_read)
    output = tmp_path / "projected"

    with pytest.raises(
        TaskBoardProjectorError, match="snapshot_binding_changed_during_read"
    ):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert swapped is True
    assert not output.exists()


def test_projector_rejects_manifest_schema_snapshot_change_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_snapshot(tmp_path)
    original = projector_module._read_bytes_snapshot
    schema_path = MANIFEST_SCHEMA.resolve()
    schema_reads = 0

    def swap_manifest_schema_snapshot(path: Path, code: str) -> Any:
        nonlocal schema_reads
        snapshot = original(path, code)
        if path.resolve() == schema_path:
            schema_reads += 1
            if schema_reads == 2:
                changed = snapshot.data + b"\n"
                return projector_module._BytesSnapshot(
                    data=changed,
                    sha256=_sha_bytes(changed),
                    size=len(changed),
                )
        return snapshot

    monkeypatch.setattr(
        projector_module, "_read_bytes_snapshot", swap_manifest_schema_snapshot
    )
    output = tmp_path / "projected"

    with pytest.raises(
        TaskBoardProjectorError, match="projector_config_changed_during_read"
    ):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert schema_reads == 2
    assert not output.exists()


def test_projector_rejects_projected_file_swap_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_snapshot(tmp_path)
    original = projector_module._read_bytes_snapshot
    swapped = False

    def swap_projected_file(path: Path, code: str) -> Any:
        nonlocal swapped
        snapshot = original(path, code)
        if (
            path.name == "clean.jsonl"
            and path.parent.name == "train"
            and path.parent.parent.name.startswith(".projected.tmp-")
            and not swapped
        ):
            swapped = True
            path.write_bytes(snapshot.data + b"\n")
        return snapshot

    monkeypatch.setattr(
        projector_module, "_read_bytes_snapshot", swap_projected_file
    )
    output = tmp_path / "projected"

    with pytest.raises(
        TaskBoardProjectorError, match="projected_manifest_file_binding_invalid"
    ):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert swapped is True
    assert not output.exists()


def test_projector_rejects_split_ids_digest_drift(tmp_path: Path) -> None:
    fixture = _build_snapshot(tmp_path)
    manifest_path = fixture.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_contract"]["partitions"]["train"]["ids_sha256"] = _sha_text(
        "wrong-instance"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha = _sha_file(manifest_path)
    (fixture.root / "manifest.json.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    output = tmp_path / "projected"

    with pytest.raises(TaskBoardProjectorError, match="train_split_ids_sha256_mismatch"):
        project_taskboards(CONFIG, fixture.root, manifest_sha, output)

    assert not output.exists()


def test_projector_rejects_credential_key_value_pair(tmp_path: Path) -> None:
    def inject_password(split: str, stage: str, row: dict[str, Any]) -> None:
        if split == "train" and stage == "planner":
            row["input"]["password"] = "correct-horse-battery-staple"

    fixture = _build_snapshot(tmp_path, mutate=inject_password)
    output = tmp_path / "projected"

    with pytest.raises(TaskBoardProjectorError, match="train_gold_record_invalid"):
        project_taskboards(CONFIG, fixture.root, fixture.manifest_sha256, output)

    assert not output.exists()


def test_projector_rejects_cross_split_task_and_secret_content(
    tmp_path: Path,
) -> None:
    overlap = _build_snapshot(tmp_path / "overlap", cross_split_task=True)
    with pytest.raises(TaskBoardProjectorError, match="source_task_cross_split"):
        project_taskboards(
            CONFIG,
            overlap.root,
            overlap.manifest_sha256,
            tmp_path / "overlap-output",
        )

    secret = _build_snapshot(tmp_path / "secret", secret=True)
    with pytest.raises(TaskBoardProjectorError, match="train_task_bank_invalid"):
        project_taskboards(
            CONFIG,
            secret.root,
            secret.manifest_sha256,
            tmp_path / "secret-output",
        )


def test_projector_rejects_output_overlap_or_existing_directory(
    tmp_path: Path,
) -> None:
    fixture = _build_snapshot(tmp_path)
    before = _source_inventory(fixture)

    with pytest.raises(TaskBoardProjectorError, match="overlaps_input"):
        project_taskboards(
            CONFIG,
            fixture.root,
            fixture.manifest_sha256,
            fixture.root / "projection",
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(TaskBoardProjectorError, match="overlaps_input"):
        project_taskboards(
            CONFIG, fixture.root, fixture.manifest_sha256, existing
        )

    assert _source_inventory(fixture) == before
