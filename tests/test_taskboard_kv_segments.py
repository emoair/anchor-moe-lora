from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from anchor_mvp.research.query_specialization import (
    ROLES,
    build_training_view,
    parse_taskboard_sidecar,
)
from anchor_mvp.research.taskboard_kv_segments import (
    FROZEN_PRODUCER_CONTRACT_SHA256,
    FROZEN_CONSUMER_CONFIG_SHA256,
    INDEX_SCHEMA_VERSION,
    TaskBoardKVSegmentError,
    load_authenticated_taskboard_kv_dataset,
    place_expert_output,
    project_taskboard_kv_segment_plans,
    validate_index_mapping,
    validate_native_sidecar_dataset,
    validate_plan_mapping,
    validate_plan_schema,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/research/taskboard_projector"
SCRIPT = ROOT / "scripts/research/materialize_taskboard_kv_segments.py"
PRODUCER_CONFIG = ROOT / "configs/research/swebench_taskboard_projector_v2.yaml"
MANIFEST_SCHEMA = ROOT / "configs/research/taskboard_projector_manifest.schema.json"
SIDECAR_SCHEMA = ROOT / "configs/research/taskboard_projector_sidecar.schema.json"
SEGMENT_SCHEMA = ROOT / "configs/research/hierarchical_task_kv_segment_plan.schema.json"
CONSUMER_CONFIG = ROOT / "configs/research/hierarchical_task_kv_mvp.yaml"
PARTITIONS = (
    ("train/clean.jsonl", "train", "clean"),
    ("train/noisy.jsonl", "train", "noisy"),
    ("calibration/clean.jsonl", "calibration", "clean"),
)
HARD_MAX_FILE_BYTES = 50_000_000
FROZEN_FIXTURE_MANIFEST_SHA256 = (
    "595cd150845015f3723e28a6aa0cb48730cdca6457580ad66a393ef4143fa2ac"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, _, _ in PARTITIONS:
        rows.extend(
            json.loads(line)
            for line in (FIXTURE / relative).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _sidecars():
    config_sha = _sha(PRODUCER_CONFIG)
    sidecar_schema_sha = _sha(SIDECAR_SCHEMA)
    segment_schema_sha = _sha(SEGMENT_SCHEMA)
    result = []
    for relative, split, variant in PARTITIONS:
        for line_number, line in enumerate(
            (FIXTURE / relative).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip():
                result.append(
                    parse_taskboard_sidecar(
                        json.loads(line),
                        source=f"{relative}:{line_number}",
                        expected_split=split,
                        expected_variant=variant,
                        expected_config_sha256=config_sha,
                        expected_sidecar_schema_sha256=sidecar_schema_sha,
                        expected_segment_plan_schema_sha256=segment_schema_sha,
                    )
                )
    return tuple(result)


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_authenticated_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Repair only temporary producer signatures, never row payloads."""

    dataset = tmp_path / "producer"
    shutil.copytree(FIXTURE, dataset)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["producer"]["manifest_schema_sha256"] = _sha(MANIFEST_SCHEMA)
    manifest["hierarchical_task_kv"]["segment_plan_location"] = (
        "outer_sidecar.segment_plan"
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.with_name("manifest.json.sha256").write_text(
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json\n",
        encoding="ascii",
        newline="",
    )

    config = yaml.safe_load(CONSUMER_CONFIG.read_text(encoding="utf-8"))
    consumer = tmp_path / "hierarchical_task_kv_mvp.yaml"
    consumer.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return dataset, consumer


def test_consumes_all_15_native_plans_without_a_second_plan_representation() -> None:
    sidecars = _sidecars()
    summary = validate_native_sidecar_dataset(sidecars)
    indexes = project_taskboard_kv_segment_plans(sidecars)

    assert len(sidecars) == len(indexes) == 15
    assert summary["task_bundles"] == 2
    assert summary["all_five_role_views_same_split"] is True
    assert summary["canonical_native_plans_consumed"] is True
    assert {index.split for index in indexes} == {"train", "calibration"}

    by_id = {sidecar.record_id: sidecar for sidecar in sidecars}
    for index in indexes:
        sidecar = by_id[index.source_sidecar_record_id]
        native_plan = sidecar.segment_plan
        assert validate_plan_mapping(
            native_plan, outer=sidecar, record=sidecar.training_record
        ) == _canonical_sha(native_plan)
        emitted = index.to_dict()
        validate_index_mapping(emitted)
        assert emitted["schema_version"] == INDEX_SCHEMA_VERSION
        assert emitted["canonical_segment_plan_sha256"] == _canonical_sha(
            native_plan
        )
        assert emitted["cache_reuse_allowed"] is False
        assert "segments" not in emitted
        assert "bindings" not in emitted
        assert "segment_plan" not in emitted
        assert not _contains_key(emitted, "content")


def test_real_training_view_excludes_forbidden_current_future_and_target_answer() -> None:
    for sidecar in _sidecars():
        record = sidecar.training_record
        plan = sidecar.segment_plan
        view = build_training_view(record)
        prompt = json.loads(view.prompt)
        target = json.loads(record.target_output)
        planned_ids = [segment["source_block_id"] for segment in plan["segments"]]
        forbidden = set(record.targets.forbidden)

        assert forbidden
        assert forbidden.issubset({block.block_id for block in record.blocks})
        assert planned_ids == list(view.visible_block_ids)
        assert planned_ids == [block["id"] for block in prompt["blocks"]]
        assert forbidden.isdisjoint(planned_ids)
        assert target["answer"] not in view.prompt
        assert plan["target_delta_policy"]["current_target_segment_emitted"] is False
        assert plan["shared_prefix_policy"][
            "forbidden_current_future_preinsert_allowed"
        ] is False


@pytest.mark.parametrize(
    "binding",
    [
        "task_bundle_sha256",
        "task_id",
        "base_task_board_sha256",
        "projector_version",
        "config_sha256",
        "sidecar_schema_sha256",
        "segment_plan_schema_sha256",
        "source_gold_sha256",
        "source_gold_file_sha256",
        "source_snapshot_sha256",
        "source_snapshot_manifest_sha256",
        "split",
        "stage",
        "expert",
        "variant",
    ],
)
def test_every_native_binding_is_cross_checked_to_outer_sidecar(binding: str) -> None:
    sidecar = _sidecars()[0]
    changed = copy.deepcopy(sidecar.segment_plan)
    changed["bindings"][binding] = (
        "0" * 64
        if binding.endswith("sha256")
        else "tampered"
    )
    with pytest.raises(TaskBoardKVSegmentError):
        validate_plan_mapping(changed, outer=sidecar, record=sidecar.training_record)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda plan, _: plan["cache_compatibility"].__setitem__("status", "exact"), "cache_compatibility"),
        (lambda plan, _: plan["cache_compatibility"].__setitem__("cache_reuse_allowed", True), "cache_compatibility"),
        (lambda plan, _: plan["target_delta_policy"].__setitem__("current_target_segment_emitted", True), "target_delta_policy"),
        (lambda plan, _: plan["shared_prefix_policy"].__setitem__("forbidden_current_future_preinsert_allowed", True), "shared_prefix_policy"),
        (lambda plan, _: plan["segments"][0].__setitem__("content_sha256", "0" * 64), "segment_id"),
        (lambda plan, _: plan["segments"][0].__setitem__("segment_id", "task-kv-segment-v1:" + "0" * 64), "segment_id"),
        (lambda plan, _: plan["segments"][1].__setitem__("dependencies", []), "dependencies"),
        (lambda plan, _: plan["segments"][1].__setitem__("parent_segment_id", None), "parent_segment_id"),
        (lambda plan, _: plan["segments"][0].__setitem__("prefix_lineage_sha256", "0" * 64), "prefix_lineage"),
        (lambda plan, record: plan["segments"][0].__setitem__("source_block_id", record.targets.forbidden[0]), "segment_id"),
    ],
)
def test_native_plan_tampering_fails_closed(mutation, match: str) -> None:
    sidecar = _sidecars()[0]
    changed = copy.deepcopy(sidecar.segment_plan)
    mutation(changed, sidecar.training_record)
    with pytest.raises(TaskBoardKVSegmentError, match=match):
        validate_plan_mapping(changed, outer=sidecar, record=sidecar.training_record)


def test_private_delta_visibility_and_scope_are_fail_closed() -> None:
    sidecar = next(row for row in _sidecars() if row.variant == "noisy")
    changed = copy.deepcopy(sidecar.segment_plan)
    private = changed["segments"][-1]
    assert private["cache_scope"] == "expert_private_delta"
    private["visibility"] = list(ROLES)
    with pytest.raises(TaskBoardKVSegmentError, match="private-delta"):
        validate_plan_mapping(changed, outer=sidecar, record=sidecar.training_record)


def test_five_role_bundle_cannot_be_partially_consumed() -> None:
    sidecars = _sidecars()
    with pytest.raises(TaskBoardKVSegmentError, match="exactly five role views"):
        validate_native_sidecar_dataset(sidecars[:-1])


def test_checked_in_native_schema_is_closed_and_authenticated() -> None:
    digest = validate_plan_schema(SEGMENT_SCHEMA)
    assert digest == _sha(SEGMENT_SCHEMA)
    changed = json.loads(SEGMENT_SCHEMA.read_text(encoding="utf-8"))
    changed["properties"]["schema_version"]["const"] = "wrong"
    # Mapping validation itself remains independently strict.
    with pytest.raises(TaskBoardKVSegmentError, match="fixed execution"):
        plan = copy.deepcopy(_sidecars()[0].segment_plan)
        plan["schema_version"] = "wrong"
        validate_plan_mapping(plan)


def test_final_producer_contract_and_fixture_signatures_are_frozen() -> None:
    assert FROZEN_PRODUCER_CONTRACT_SHA256 == {
        "producer_config": _sha(PRODUCER_CONFIG),
        "manifest_schema": _sha(MANIFEST_SCHEMA),
        "sidecar_schema": _sha(SIDECAR_SCHEMA),
        "segment_plan_schema": _sha(SEGMENT_SCHEMA),
    }
    manifest_sha = _sha(FIXTURE / "manifest.json")
    assert manifest_sha == FROZEN_FIXTURE_MANIFEST_SHA256
    assert (FIXTURE / "manifest.json.sha256").read_bytes() == (
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    assert FROZEN_CONSUMER_CONFIG_SHA256 == _sha(CONSUMER_CONFIG)


def test_new_output_is_private_until_exact_downstream_commit() -> None:
    content = "synthetic expert output"
    owner = "frontend_gen"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    private = place_expert_output(content=content, owner_expert_id=owner)
    assert private.cache_scope == "expert_private_delta"
    assert private.downstream_expert_ids == (owner,)
    assert private.explicitly_committed is False

    commit = {
        "schema_version": "anchor.taskboard-kv-explicit-commit.v1",
        "commit_id": "synthetic-commit-1",
        "committed": True,
        "approved_cache_scope": "downstream_task_shared_immutable",
        "owner_expert_id": owner,
        "content_sha256": digest,
        "downstream_expert_ids": ["frontend_review", "security_gate"],
    }
    promoted = place_expert_output(
        content=content, owner_expert_id=owner, commit_metadata=commit
    )
    assert promoted.cache_scope == "downstream_task_shared_immutable"
    assert promoted.explicitly_committed is True

    with pytest.raises(TaskBoardKVSegmentError, match="exactly bind"):
        place_expert_output(
            content=content,
            owner_expert_id=owner,
            commit_metadata=dict(commit, content_sha256="0" * 64),
        )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _cli_contract_args(dataset: Path, consumer: Path) -> tuple[str, ...]:
    return (
        "--input-root",
        str(dataset),
        "--consumer-config",
        str(consumer),
        "--expected-consumer-config-sha256",
        _sha(consumer),
    )


def test_authenticated_loader_consumes_config_and_all_15_fixture_rows(
    tmp_path: Path,
) -> None:
    dataset, consumer = _stage_authenticated_fixture(tmp_path)
    sidecars, _, validation = load_authenticated_taskboard_kv_dataset(
        dataset,
        manifest_path=dataset / "manifest.json",
        producer_config_path=PRODUCER_CONFIG,
        manifest_schema_path=MANIFEST_SCHEMA,
        sidecar_schema_path=SIDECAR_SCHEMA,
        segment_plan_schema_path=SEGMENT_SCHEMA,
        consumer_config_path=consumer,
        expected_consumer_config_sha256=_sha(consumer),
    )
    assert len(sidecars) == 15
    assert validation["consumer_config_sha256"] == _sha(consumer)
    assert validation["producer_config_sha256"] == _sha(PRODUCER_CONFIG)
    assert validation["producer_manifest_schema_sha256"] == _sha(MANIFEST_SCHEMA)
    assert validation["producer_sidecar_schema_sha256"] == _sha(SIDECAR_SCHEMA)
    assert validation["producer_segment_plan_schema_sha256"] == _sha(SEGMENT_SCHEMA)


def test_final_checked_in_fixture_loads_without_test_resigning() -> None:
    sidecars, _, validation = load_authenticated_taskboard_kv_dataset(
        FIXTURE,
        manifest_path=FIXTURE / "manifest.json",
        producer_config_path=PRODUCER_CONFIG,
        manifest_schema_path=MANIFEST_SCHEMA,
        sidecar_schema_path=SIDECAR_SCHEMA,
        segment_plan_schema_path=SEGMENT_SCHEMA,
        consumer_config_path=CONSUMER_CONFIG,
        expected_consumer_config_sha256=_sha(CONSUMER_CONFIG),
    )
    assert len(sidecars) == 15
    assert validation["source_manifest_sha256"] == FROZEN_FIXTURE_MANIFEST_SHA256


def test_cli_dry_run_writes_nothing_and_reports_content_free_index(
    tmp_path: Path,
) -> None:
    dataset, consumer = _stage_authenticated_fixture(tmp_path)
    output = tmp_path / "dry-output"
    result = _run_cli(
        *_cli_contract_args(dataset, consumer),
        "--dry-run",
        "--output-root",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    manifest = response["manifest"]
    assert response["mode"] == "dry_run"
    assert manifest["index_summary"]["records"] == 15
    assert manifest["consumer_contract"]["config_sha256"] == _sha(consumer)
    assert manifest["canonical_segment_plan_written"] is False
    assert manifest["source_body_written"] is False
    assert manifest["provider_requests"] == 0
    assert manifest["model_loaded"] is False
    assert manifest["gpu_used"] is False
    assert not output.exists()
    assert not _contains_key(response, "content")


def test_cli_execute_is_deterministic_and_every_file_is_below_50mb(
    tmp_path: Path,
) -> None:
    dataset, consumer = _stage_authenticated_fixture(tmp_path)
    output = tmp_path / "materialized"
    args = (
        *_cli_contract_args(dataset, consumer),
        "--execute",
        "--output-root",
        str(output),
    )
    first = _run_cli(*args)
    assert first.returncode == 0, first.stderr
    output_dir = Path(json.loads(first.stdout)["output_dir"])
    assert output_dir.is_dir()

    records = 0
    for path in output_dir.rglob("*"):
        if path.is_file():
            assert path.stat().st_size < HARD_MAX_FILE_BYTES
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                emitted = json.loads(line)
                validate_index_mapping(emitted)
                assert "segments" not in emitted
                assert "segment_plan" not in emitted
                assert not _contains_key(emitted, "content")
                records += 1
    assert records == 15

    manifest_bytes = (output_dir / "manifest.json").read_bytes()
    declaration = (output_dir / "manifest.json.sha256").read_bytes()
    assert declaration == (
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json\n".encode(
            "ascii"
        )
    )
    second = _run_cli(*args)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["output_dir"] == str(output_dir)


@pytest.mark.parametrize(
    "tamper",
    [
        "partition",
        "manifest_sidecar",
        "consumer",
        "consumer_path",
        "segment_schema",
    ],
)
def test_cli_authentication_tampering_fails_closed(
    tmp_path: Path, tamper: str
) -> None:
    dataset, consumer = _stage_authenticated_fixture(tmp_path)
    extra: tuple[str, ...] = ()
    if tamper == "partition":
        path = dataset / "train/clean.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    elif tamper == "manifest_sidecar":
        (dataset / "manifest.json.sha256").write_text(
            f"{'0' * 64}  manifest.json\n", encoding="ascii", newline=""
        )
    elif tamper == "consumer":
        value = yaml.safe_load(consumer.read_text(encoding="utf-8"))
        value["claim_scope"] = "not-authenticated"
        consumer.write_text(yaml.safe_dump(value), encoding="utf-8")
    elif tamper == "consumer_path":
        value = yaml.safe_load(consumer.read_text(encoding="utf-8"))
        value["source_of_truth"]["segment_plan_schema"]["path"] = (
            "configs/research/not-the-authenticated-schema.json"
        )
        consumer.write_text(yaml.safe_dump(value), encoding="utf-8")
    else:
        changed_schema = tmp_path / "changed-segment.schema.json"
        changed_schema.write_bytes(SEGMENT_SCHEMA.read_bytes() + b"\n")
        extra = ("--segment-plan-schema", str(changed_schema))
    result = _run_cli(
        *_cli_contract_args(dataset, consumer),
        *extra,
        "--dry-run",
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error:")


def test_cli_rejects_a_shard_ceiling_above_authenticated_config(
    tmp_path: Path,
) -> None:
    dataset, consumer = _stage_authenticated_fixture(tmp_path)
    result = _run_cli(
        *_cli_contract_args(dataset, consumer),
        "--dry-run",
        "--max-shard-bytes",
        str(HARD_MAX_FILE_BYTES),
    )
    assert result.returncode == 2
    assert "below 50,000,000" in result.stderr
