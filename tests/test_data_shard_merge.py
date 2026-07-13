from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from anchor_mvp.data.pipeline import DistillationPipeline
from anchor_mvp.data.shard_merge import (
    FILES,
    ShardMergeError,
    merge_distillation_shards,
)
from anchor_mvp.data.teacher import MockTeacher


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _fixture_sources(
    tmp_path: Path, *, seed_count: int = 2
) -> tuple[Path, Path, list[str]]:
    if seed_count < 2:
        raise ValueError("fixture requires at least two seeds")
    generated = tmp_path / "generated-fixture"
    report = asyncio.run(
        DistillationPipeline(
            teacher=MockTeacher(),
            sop_dir=ROOT / "skills",
            output_dir=generated,
            concurrency=2,
        ).run(seed_count=seed_count)
    )
    assert report.errors == ()
    seeds = _jsonl(generated / "seeds.jsonl")
    seed_ids = [str(row["seed_id"]) for row in seeds]
    all_rows = {filename: _jsonl(generated / filename) for filename in FILES}
    plan0 = next(
        row
        for row in all_rows["data_plan.jsonl"]
        if row["provenance"]["seed_id"] == seed_ids[0]
    )
    plan0["provenance"]["source_offset"] = {
        "source_index": 0,
        "line_offset": 7,
    }
    plan0["provenance"]["teacher"]["provider"]["fixture_marker"] = (
        "provider-provenance-preserved"
    )

    base = tmp_path / "base"
    shard = tmp_path / "shard-1"
    for filename in FILES:
        rows = all_rows[filename]
        if filename == "seeds.jsonl":
            by_seed = {str(row["seed_id"]): row for row in rows}
        else:
            by_seed = {str(row["provenance"]["seed_id"]): row for row in rows}
        first = json.loads(json.dumps(by_seed[seed_ids[0]]))
        remaining = [
            json.loads(json.dumps(by_seed[seed_id])) for seed_id in seed_ids[1:]
        ]
        _write_jsonl(base / filename, [first])
        # Include one exact base duplicate in every file to exercise all
        # deterministic duplicate counters without reading any live corpus.
        _write_jsonl(shard / filename, [first, *remaining])
    return base, shard, seed_ids


def _row_in(rows: list[dict], seed_id: str) -> dict:
    return next(
        row
        for row in rows
        if row.get("seed_id") == seed_id
        or row.get("provenance", {}).get("seed_id") == seed_id
    )


def _row_for_seed(path: Path, seed_id: str) -> dict:
    return _row_in(_jsonl(path), seed_id)


def test_atomic_merge_dry_run_deduplicates_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path)
    target = tmp_path / "merged"

    dry_manifest = merge_distillation_shards(
        base_dir=base,
        shard_dirs=[shard],
        target_dir=target,
        dry_run=True,
    )

    assert not target.exists()
    assert dry_manifest["counts"]["output_rows"] == {filename: 2 for filename in FILES}
    assert dry_manifest["duplicates"]["eliminated_total"] == len(FILES)
    assert set(dry_manifest) == {
        "schema_version",
        "counts",
        "hashes",
        "duplicates",
        "lineage",
        "strict_gold",
        "task_cards",
    }
    assert dry_manifest["schema_version"] == "anchor.distillation-shard-merge.v2"
    assert dry_manifest["lineage"] == {
        "seed_count": 2,
        "stage_counts": {
            task: 2
            for task in (
                "plan",
                "tool_policy",
                "frontend",
                "review",
                "security",
            )
        },
        "partial_chain_distribution": {
            "seed_only": 0,
            "through_plan": 0,
            "through_tool_policy": 0,
            "through_frontend": 0,
            "through_review": 0,
        },
        "partial_chain_count": 0,
        "raw_complete_chain_intersection": {
            "count": 2,
            "seed_ids_sha256": dry_manifest["lineage"][
                "raw_complete_chain_intersection"
            ]["seed_ids_sha256"],
        },
    }
    assert dry_manifest["strict_gold"] == {
        "decision": "deferred_to_partition",
        "partition_required": True,
    }
    assert dry_manifest["task_cards"]["card_count"] == 2

    manifest = merge_distillation_shards(
        base_dir=base,
        shard_dirs=[shard],
        target_dir=target,
    )

    assert manifest == dry_manifest
    assert target.is_dir()
    assert set(path.name for path in target.iterdir()) == {*FILES, "manifest.json"}
    assert all(len(_jsonl(target / filename)) == 2 for filename in FILES)
    merged_plan = _row_for_seed(target / "data_plan.jsonl", seed_ids[0])
    assert merged_plan["provenance"]["source_offset"] == {
        "source_index": 0,
        "line_offset": 7,
    }
    assert merged_plan["provenance"]["teacher"]["model"] == "mock-teacher-v1"
    assert (
        merged_plan["provenance"]["teacher"]["provider"]["fixture_marker"]
        == "provider-provenance-preserved"
    )
    manifest_text = (target / "manifest.json").read_text(encoding="utf-8")
    assert "Build an accessible" not in manifest_text
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(ShardMergeError, match="overwrite refused"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=target,
        )
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_manifest_reports_every_partial_chain_depth_without_promoting_gold(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path, seed_count=6)
    retained_depth = {
        seed_ids[0]: 5,
        seed_ids[1]: 0,
        seed_ids[2]: 1,
        seed_ids[3]: 2,
        seed_ids[4]: 3,
        seed_ids[5]: 4,
    }
    task_depth = {
        "data_plan.jsonl": 1,
        "data_tool_policy.jsonl": 2,
        "data_frontend.jsonl": 3,
        "data_review.jsonl": 4,
        "data_security.jsonl": 5,
    }
    for filename, depth in task_depth.items():
        path = shard / filename
        _write_jsonl(
            path,
            [
                row
                for row in _jsonl(path)
                if retained_depth[str(row["provenance"]["seed_id"])] >= depth
            ],
        )

    target = tmp_path / "partial-depth-target"
    manifest = merge_distillation_shards(
        base_dir=base,
        shard_dirs=[shard],
        target_dir=target,
        dry_run=True,
    )

    assert not target.exists()
    assert manifest["lineage"]["stage_counts"] == {
        "plan": 5,
        "tool_policy": 4,
        "frontend": 3,
        "review": 2,
        "security": 1,
    }
    assert manifest["lineage"]["partial_chain_distribution"] == {
        "seed_only": 1,
        "through_plan": 1,
        "through_tool_policy": 1,
        "through_frontend": 1,
        "through_review": 1,
    }
    complete = manifest["lineage"]["raw_complete_chain_intersection"]["count"]
    assert complete == 1
    assert manifest["lineage"]["partial_chain_count"] + complete == 6
    assert manifest["strict_gold"] == {
        "decision": "deferred_to_partition",
        "partition_required": True,
    }


def test_same_record_id_with_different_content_fails_without_publish(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path)
    path = shard / "data_frontend.jsonl"
    rows = _jsonl(path)
    record = _row_in(rows, seed_ids[0])
    record["output"]["code"] += "\n// bounded fixture change"
    record["messages"][-1]["content"] = record["output"]["code"].strip()
    _write_jsonl(path, rows)
    target = tmp_path / "must-not-publish"

    with pytest.raises(ShardMergeError, match="same record id"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=target,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".must-not-publish.merge-*"))


def test_same_task_seed_with_two_records_fails_closed(tmp_path: Path) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path)
    path = shard / "data_plan.jsonl"
    rows = _jsonl(path)
    alternate = json.loads(json.dumps(_row_for_seed(path, seed_ids[1])))
    alternate["id"] = "record_alternate_for_same_seed"
    alternate["output"]["summary"] += " Alternate."
    alternate["messages"][-1]["content"] = json.dumps(
        alternate["output"], ensure_ascii=False, sort_keys=True
    )
    rows.append(alternate)
    _write_jsonl(path, rows)

    with pytest.raises(ShardMergeError, match="multiple records for one seed"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "merged",
        )


def test_unresolved_and_cross_seed_lineage_fail_closed(tmp_path: Path) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path)
    frontend_path = shard / "data_frontend.jsonl"
    rows = _jsonl(frontend_path)
    frontend = _row_in(rows, seed_ids[1])
    frontend["provenance"]["source_plan_record_id"] = "missing-plan-record"
    _write_jsonl(frontend_path, rows)
    with pytest.raises(ShardMergeError, match="does not resolve"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "missing-target",
        )

    base, shard, seed_ids = _fixture_sources(tmp_path / "cross-seed")
    frontend_path = shard / "data_frontend.jsonl"
    rows = _jsonl(frontend_path)
    frontend = _row_in(rows, seed_ids[1])
    base_plan = _row_for_seed(base / "data_plan.jsonl", seed_ids[0])
    frontend["provenance"]["source_plan_record_id"] = base_plan["id"]
    _write_jsonl(frontend_path, rows)
    with pytest.raises(ShardMergeError, match="crosses seed lineage"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "cross-target",
        )


@pytest.mark.parametrize(
    ("filename", "source_field"),
    [
        ("data_tool_policy.jsonl", "source_plan_record_id"),
        ("data_frontend.jsonl", "source_plan_record_id"),
        ("data_frontend.jsonl", "source_tool_policy_record_id"),
        ("data_review.jsonl", "source_frontend_record_id"),
        ("data_security.jsonl", "source_review_record_id"),
    ],
)
def test_every_declared_dependency_rejects_an_unresolved_source_id(
    tmp_path: Path, filename: str, source_field: str
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path / f"{filename}-{source_field}")
    path = shard / filename
    rows = _jsonl(path)
    record = _row_in(rows, seed_ids[1])
    record["provenance"][source_field] = "missing-upstream-record"
    _write_jsonl(path, rows)

    with pytest.raises(ShardMergeError, match="source reference does not resolve"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "unresolved-edge-target",
            dry_run=True,
        )


def test_canonical_assistant_and_ark_credential_fail_without_echo(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path / "assistant")
    path = shard / "data_plan.jsonl"
    rows = _jsonl(path)
    _row_in(rows, seed_ids[1])["messages"][-1]["content"] = "not canonical"
    _write_jsonl(path, rows)
    with pytest.raises(ShardMergeError, match="canonical dataset validation failed"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "assistant-target",
        )

    base, shard, seed_ids = _fixture_sources(tmp_path / "secret")
    path = shard / "data_frontend.jsonl"
    rows = _jsonl(path)
    frontend = _row_in(rows, seed_ids[1])
    credential = "ark-fixturecredential1234567890"
    frontend["output"]["code"] += f"\n// {credential}"
    frontend["messages"][-1]["content"] = frontend["output"]["code"].strip()
    _write_jsonl(path, rows)
    with pytest.raises(ShardMergeError, match="credential-like") as captured:
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "secret-target",
        )
    assert credential not in str(captured.value)
    assert not (tmp_path / "secret-target").exists()


def test_seed_fingerprint_alias_and_record_fingerprint_alias_are_rejected(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path / "seed-alias")
    seed_path = shard / "seeds.jsonl"
    rows = _jsonl(seed_path)
    first = _row_for_seed(base / "seeds.jsonl", seed_ids[0])
    second = _row_in(rows, seed_ids[1])
    second["request"] = first["request"]
    _write_jsonl(seed_path, rows)
    with pytest.raises(
        ShardMergeError, match="fingerprint maps|task-card binding is invalid"
    ):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "seed-alias-target",
        )

    base, shard, seed_ids = _fixture_sources(tmp_path / "record-alias")
    plan_path = shard / "data_plan.jsonl"
    rows = _jsonl(plan_path)
    duplicate = json.loads(
        json.dumps(_row_for_seed(base / "data_plan.jsonl", seed_ids[0]))
    )
    duplicate["id"] = "record_fingerprint_alias"
    rows.append(duplicate)
    _write_jsonl(plan_path, rows)
    with pytest.raises(ShardMergeError, match="fingerprint maps"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "record-alias-target",
        )


def test_partial_funnel_and_preexisting_publish_lock_are_handled(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path / "incomplete")
    security_path = shard / "data_security.jsonl"
    rows = [
        row
        for row in _jsonl(security_path)
        if row["provenance"]["seed_id"] != seed_ids[1]
    ]
    _write_jsonl(security_path, rows)
    partial_target = tmp_path / "incomplete-target"
    manifest = merge_distillation_shards(
        base_dir=base,
        shard_dirs=[shard],
        target_dir=partial_target,
        dry_run=True,
    )
    assert not partial_target.exists()
    assert manifest["lineage"]["stage_counts"] == {
        "plan": 2,
        "tool_policy": 2,
        "frontend": 2,
        "review": 2,
        "security": 1,
    }
    assert manifest["lineage"]["partial_chain_distribution"] == {
        "seed_only": 0,
        "through_plan": 0,
        "through_tool_policy": 0,
        "through_frontend": 0,
        "through_review": 1,
    }
    assert manifest["lineage"]["partial_chain_count"] == 1
    assert manifest["lineage"]["raw_complete_chain_intersection"]["count"] == 1

    base, shard, _ = _fixture_sources(tmp_path / "locked")
    target = tmp_path / "locked-target"
    lock_path = target.parent / f".{target.name}.merge.lock"
    lock_bytes = b"fixture-owned-by-another-process"
    lock_path.write_bytes(lock_bytes)
    with pytest.raises(ShardMergeError, match="already publishing"):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=target,
        )
    assert lock_path.read_bytes() == lock_bytes
    assert not target.exists()


def test_partial_funnel_still_requires_every_existing_dependency(
    tmp_path: Path,
) -> None:
    base, shard, seed_ids = _fixture_sources(tmp_path)
    policy_path = shard / "data_tool_policy.jsonl"
    _write_jsonl(
        policy_path,
        [
            row
            for row in _jsonl(policy_path)
            if row["provenance"]["seed_id"] != seed_ids[1]
        ],
    )

    with pytest.raises(
        ShardMergeError, match="dependency closure|source reference does not resolve"
    ):
        merge_distillation_shards(
            base_dir=base,
            shard_dirs=[shard],
            target_dir=tmp_path / "orphan-target",
            dry_run=True,
        )
