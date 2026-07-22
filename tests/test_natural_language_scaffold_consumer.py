from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import natural_language_scaffold_consumer as consumer_module
from anchor_mvp.research.natural_language_scaffold_consumer import (
    CONSUMER_CONFIG_SHA256,
    CONFIG_SHA256,
    FIXTURE_MANIFEST_SHA256,
    FIXTURE_MANIFEST_SIDECAR_SHA256,
    MANIFEST_SCHEMA_SHA256,
    RECORD_SCHEMA_SHA256,
    SMOKE_CONTRACT_SHA256,
    SMOKE_SCHEMA_SHA256,
    AuthenticatedScaffoldRecord,
    BoundScaffold,
    NaturalLanguageScaffoldConsumerError,
    bind_scaffolds_to_taskboard,
    build_bound_scaffold_view,
    build_contract_ablation_view,
    load_natural_language_scaffold_fixture,
    paired_ablation_summary,
    paired_bound_ablation_summary,
    planner_scaffold_from_record,
    validate_scaffold_record,
    validate_two_request_gate,
)
from anchor_mvp.research.query_specialization import build_training_view


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/research/swebench_natural_language_scaffold"
TASKBOARD = ROOT / "fixtures/research/taskboard_projector"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "scaffold"
    shutil.copytree(FIXTURE, destination)
    return destination


def _copy_contract_root(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    destination = root / "configs/research"
    destination.mkdir(parents=True)
    for name in (
        "natural_language_scaffold_consumer_v1.yaml",
        "swebench_natural_language_scaffold_v1.yaml",
        "swebench_natural_language_scaffold_sidecar.schema.json",
        "swebench_natural_language_scaffold_manifest.schema.json",
        "swebench_natural_language_scaffold_smoke_contract.schema.json",
        "swebench_natural_language_scaffold_smoke_v1.yaml",
    ):
        shutil.copy2(ROOT / "configs/research" / name, destination / name)
    return root


def _fixture():
    return load_natural_language_scaffold_fixture(FIXTURE, repo_root=ROOT)


def _mutable_record(record):
    return json.loads(consumer_module._canonical_bytes(record).decode("utf-8"))


def test_frozen_contract_physical_hashes_are_exact() -> None:
    expected = {
        "configs/research/natural_language_scaffold_consumer_v1.yaml": CONSUMER_CONFIG_SHA256,
        "configs/research/swebench_natural_language_scaffold_v1.yaml": CONFIG_SHA256,
        "configs/research/swebench_natural_language_scaffold_sidecar.schema.json": RECORD_SCHEMA_SHA256,
        "configs/research/swebench_natural_language_scaffold_manifest.schema.json": MANIFEST_SCHEMA_SHA256,
        "configs/research/swebench_natural_language_scaffold_smoke_contract.schema.json": SMOKE_SCHEMA_SHA256,
        "configs/research/swebench_natural_language_scaffold_smoke_v1.yaml": SMOKE_CONTRACT_SHA256,
        "fixtures/research/swebench_natural_language_scaffold/manifest.json": FIXTURE_MANIFEST_SHA256,
    }
    assert {relative: _sha(ROOT / relative) for relative in expected} == expected


def test_bilingual_preflight_commands_are_identical_and_hash_locked() -> None:
    def powershell_block(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return text.split("```powershell\n", 1)[1].split("\n```", 1)[0]

    english = powershell_block(ROOT / "docs/rfcs/natural_language_scaffold_consumer.md")
    chinese = powershell_block(
        ROOT / "docs/rfcs/natural_language_scaffold_consumer.zh-CN.md"
    )
    assert english == chinese
    assert "--expected-consumer-config-sha256" in english
    assert CONSUMER_CONFIG_SHA256 in english


def test_published_draft202012_schema_accepts_all_twenty_records() -> None:
    schema = json.loads(
        (
            ROOT
            / "configs/research/swebench_natural_language_scaffold_sidecar.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    accepted = 0
    for relative in (
        "train/json_only.jsonl",
        "train/concise_rationale_plus_json.jsonl",
        "calibration/json_only.jsonl",
        "calibration/concise_rationale_plus_json.jsonl",
    ):
        for line_number, line in enumerate(
            (FIXTURE / relative).read_bytes().splitlines(), start=1
        ):
            record = json.loads(line)
            error = next(validator.iter_errors(record), None)
            if error is not None:
                pytest.fail(
                    "published_schema_rejected_authenticated_record:"
                    f"{relative}:{line_number}:"
                    f"instance_path={list(error.absolute_path)}:"
                    f"schema_path={list(error.absolute_schema_path)}",
                    pytrace=False,
                )
            accepted += 1
    assert accepted == 20


def test_official_fixture_loads_and_reports_only_non_authorizing_counts() -> None:
    fixture = _fixture()
    assert fixture.summary == {
        "schema_version": "anchor.natural-language-scaffold-manifest.v1",
        "record_schema_version": "anchor.natural-language-scaffold.v1",
        "smoke_schema_version": "anchor.natural-language-scaffold-smoke-contract.v1",
        "consumer_config_sha256": CONSUMER_CONFIG_SHA256,
        "manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "manifest_sha256_sidecar_sha256": FIXTURE_MANIFEST_SIDECAR_SHA256,
        "records": 20,
        "pairs": 10,
        "task_bundles": 2,
        "training_authorized": False,
        "quality_validated": False,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
    }


def test_manifest_sha256_sidecar_is_mandatory_and_strict(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    (fixture / "manifest.json.sha256").unlink()
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_manifest_sha256_sidecar_required",
    ):
        load_natural_language_scaffold_fixture(fixture, repo_root=ROOT)

    fixture = _copy_fixture(tmp_path / "second")
    sidecar = fixture / "manifest.json.sha256"
    sidecar.write_bytes(sidecar.read_bytes().replace(b"  manifest", b" *manifest"))
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_manifest_sha256_sidecar_invalid",
    ):
        load_natural_language_scaffold_fixture(fixture, repo_root=ROOT)


def test_consumer_config_is_authoritative_and_hash_locked(tmp_path: Path) -> None:
    config = tmp_path / "consumer.yaml"
    config.write_bytes(
        (
            ROOT / "configs/research/natural_language_scaffold_consumer_v1.yaml"
        ).read_bytes()
        + b"\n"
    )
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_consumer_config_hash_invalid",
    ):
        load_natural_language_scaffold_fixture(
            FIXTURE,
            repo_root=ROOT,
            consumer_config_path=config,
            expected_consumer_config_sha256=CONSUMER_CONFIG_SHA256,
        )


def test_authenticated_records_and_manifest_are_recursively_immutable() -> None:
    fixture = _fixture()
    with pytest.raises(TypeError):
        fixture.records[0]["scaffold_text"] = "mutated"
    with pytest.raises(TypeError):
        fixture.records[0]["routing_json"]["language"] = "mutated"
    with pytest.raises(AttributeError):
        fixture.records[0]["tool_calls"].append({})
    with pytest.raises(TypeError):
        fixture.manifest["claim_scope"] = "mutated"


def test_manifest_and_partition_drift_fail_before_records_are_returned(
    tmp_path: Path,
) -> None:
    fixture = _copy_fixture(tmp_path)
    manifest = fixture / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (fixture / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii", newline="\n"
    )
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError, match="scaffold_manifest_hash_invalid"
    ):
        load_natural_language_scaffold_fixture(fixture, repo_root=ROOT)

    fixture = _copy_fixture(tmp_path / "partition")
    partition = fixture / "train/json_only.jsonl"
    partition.write_bytes(partition.read_bytes() + b"\n")
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_partition_hash_invalid",
    ):
        load_natural_language_scaffold_fixture(fixture, repo_root=ROOT)


def test_authenticated_snapshot_replacement_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _copy_fixture(tmp_path)
    original = consumer_module._read_bytes_snapshot
    changed = False

    def swap_after_snapshot(path: Path, code: str):
        nonlocal changed
        snapshot = original(path, code)
        if path.name == "manifest.json" and not changed:
            changed = True
            path.write_bytes(snapshot.data + b"\n")
        return snapshot

    monkeypatch.setattr(consumer_module, "_read_bytes_snapshot", swap_after_snapshot)
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_authenticated_snapshot_changed",
    ):
        load_natural_language_scaffold_fixture(fixture, repo_root=ROOT)


@pytest.mark.parametrize(
    "target_name",
    [
        "swebench_natural_language_scaffold_sidecar.schema.json",
        "json_only.jsonl",
    ],
)
def test_schema_and_partition_replacement_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    repository = _copy_contract_root(tmp_path)
    fixture = _copy_fixture(tmp_path)
    original = consumer_module._read_bytes_snapshot
    changed = False

    def swap_after_snapshot(path: Path, code: str):
        nonlocal changed
        snapshot = original(path, code)
        if path.name == target_name and not changed:
            changed = True
            path.write_bytes(snapshot.data + b"\n")
        return snapshot

    monkeypatch.setattr(consumer_module, "_read_bytes_snapshot", swap_after_snapshot)
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_authenticated_snapshot_changed",
    ):
        load_natural_language_scaffold_fixture(fixture, repo_root=repository)


def test_closed_record_schema_rejects_unknown_and_token_coordinate_fields() -> None:
    fixture = _fixture()
    unknown = _mutable_record(fixture.records[0])
    unknown["surprise"] = True
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_schema_instance_invalid",
    ):
        validate_scaffold_record(unknown, repo_root=ROOT)

    token_leak = _mutable_record(fixture.records[0])
    token_leak["routing_json"]["token_index"] = 1
    with pytest.raises(NaturalLanguageScaffoldConsumerError):
        validate_scaffold_record(token_leak, repo_root=ROOT)


def test_pair_ablation_has_identical_inputs_and_distinct_targets() -> None:
    fixture = _fixture()
    summary = paired_ablation_summary(fixture.records)
    assert summary["pairs"] == 10
    assert summary["paired_inputs_identical"] is True
    assert summary["targets_variant_specific"] is True
    assert summary["training_authorized"] is False


def test_every_scaffold_cross_binds_to_real_filtered_taskboard_view() -> None:
    fixture = _fixture()
    bound = bind_scaffolds_to_taskboard(fixture, TASKBOARD)
    assert len(bound) == 20
    views = [build_bound_scaffold_view(item) for item in bound]
    assert all(view["request1_candidate_only"] is True for view in views)
    assert all(view["request2_eligible"] is False for view in views)
    assert all(view["training_authorized"] is False for view in views)

    pair_inputs: dict[str, set[str]] = {}
    for record_number, (item, view) in enumerate(
        zip(bound, views, strict=True), start=1
    ):
        filtered = build_training_view(item.sidecar.training_record)
        assert hashlib.sha256(view["messages"][0]["content"].encode()).digest() == (
            hashlib.sha256(filtered.prompt.encode()).digest()
        )
        assert hashlib.sha256(view["messages"][1]["content"].encode()).digest() == (
            hashlib.sha256(item.record["scaffold_text"].encode()).digest()
        )
        target_answer = json.loads(item.sidecar.training_record.target_output)["answer"]
        if target_answer in filtered.prompt:
            pytest.fail(
                f"target_body_visible_in_filtered_prompt:{record_number}", pytrace=False
            )
        if target_answer in view["messages"][1]["content"]:
            pytest.fail(
                f"target_body_visible_in_scaffold_target:{record_number}",
                pytrace=False,
            )
        forbidden = set(item.sidecar.training_record.targets.forbidden)
        for block in item.sidecar.training_record.blocks:
            if block.block_id in forbidden and block.content in filtered.prompt:
                pytest.fail(
                    f"forbidden_body_visible_in_filtered_prompt:{record_number}",
                    pytrace=False,
                )
            if (
                block.block_id in forbidden
                and block.content
                and block.content in view["messages"][1]["content"]
            ):
                pytest.fail(
                    f"forbidden_body_visible_in_scaffold_target:{record_number}",
                    pytrace=False,
                )
        pair_inputs.setdefault(str(view["pair_id"]), set()).add(
            str(view["input_sha256"])
        )
    assert len(pair_inputs) == 10
    assert all(len(inputs) == 1 for inputs in pair_inputs.values())
    summary = paired_bound_ablation_summary(views)
    assert summary["pairs"] == 10
    assert summary["paired_inputs_identical"] is True

    tampered = [dict(view) for view in views]
    tampered[0]["input_sha256"] = "0" * 64
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_bound_ablation_invariant_invalid",
    ):
        paired_bound_ablation_summary(tampered)


def test_task_bundle_is_the_split_group_and_contains_five_roles() -> None:
    fixture = _fixture()
    groups: dict[str, list[dict]] = {}
    for record in fixture.records:
        groups.setdefault(str(record["task_bundle_sha256"]), []).append(dict(record))
    assert len(groups) == 2
    for records in groups.values():
        assert len(records) == 10
        assert len({record["split"] for record in records}) == 1
        assert {record["expert"] for record in records} == {
            "planner",
            "tool_policy",
            "frontend_gen",
            "frontend_review",
            "security_gate",
        }


def test_two_request_gate_is_request1_only_and_fail_closed_on_drift() -> None:
    fixture = _fixture()
    for record in fixture.records:
        gate = validate_two_request_gate(record)
        assert gate["request1_candidate"] is True
        assert gate["request2_eligible"] is False
        assert gate["planner_private_kv_transfer"] is False
        assert gate["training_authorized"] is False

    drift = _mutable_record(fixture.records[0])
    drift["route_boundary"]["planner_private_tail_kv_transfer_allowed"] = True
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError, match="scaffold_route_invalid"
    ):
        validate_two_request_gate(drift)


def test_authenticated_record_builds_unarmed_runtime_candidate() -> None:
    fixture = _fixture()
    bound = bind_scaffolds_to_taskboard(fixture, TASKBOARD)
    candidate = planner_scaffold_from_record(bound[0])
    assert candidate.task_bundle_sha256 == fixture.records[0]["task_bundle_sha256"]
    assert candidate.target_expert_id == fixture.records[0]["expert"]
    assert hashlib.sha256(candidate.natural_language_scaffold.encode()).digest() == (
        hashlib.sha256(fixture.records[0]["scaffold_text"].encode()).digest()
    )
    assert "private_branch_id" not in candidate.structured_plan


def test_runtime_and_materializers_reject_naked_or_forged_records() -> None:
    fixture = _fixture()
    bound = bind_scaffolds_to_taskboard(fixture, TASKBOARD)
    naked = dict(fixture.records[0])

    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_authenticated_record_required",
    ):
        build_contract_ablation_view(naked)  # type: ignore[arg-type]
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_bound_receipt_required",
    ):
        planner_scaffold_from_record(naked)  # type: ignore[arg-type]
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_bound_factory_required",
    ):
        BoundScaffold(
            record=fixture.records[0],
            sidecar=bound[0].sidecar,
            record_sha256=fixture.records[0].canonical_sha256,
            sidecar_sha256="0" * 64,
            binding_sha256="0" * 64,
            _capability=object(),
        )
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_authenticated_record_factory_required",
    ):
        AuthenticatedScaffoldRecord(
            data=fixture.records[0],
            canonical_sha256=fixture.records[0].canonical_sha256,
            manifest_sha256=fixture.records[0].manifest_sha256,
            scaffold_partition_sha256=(fixture.records[0].scaffold_partition_sha256),
            _capability=object(),
        )


def test_authenticated_and_bound_canonical_hashes_are_rechecked() -> None:
    fixture = _fixture()
    record = fixture.records[0]
    object.__setattr__(record, "canonical_sha256", "0" * 64)
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_authenticated_record_hash_invalid",
    ):
        build_contract_ablation_view(record)

    fresh = _fixture()
    bound = bind_scaffolds_to_taskboard(fresh, TASKBOARD)[0]
    object.__setattr__(bound, "binding_sha256", "0" * 64)
    with pytest.raises(
        NaturalLanguageScaffoldConsumerError,
        match="scaffold_bound_receipt_hash_invalid",
    ):
        build_bound_scaffold_view(bound)


def test_consumer_preflight_cli_is_content_free_and_model_free() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/research/preflight_natural_language_scaffold_consumer.py"
            ),
            "--expected-consumer-config-sha256",
            CONSUMER_CONFIG_SHA256,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)
    assert output["status"] == "contract_preflight_passed"
    assert output["consumer_config_sha256"] == CONSUMER_CONFIG_SHA256
    assert output["records"] == output["bound_taskboard_records"] == 20
    assert output["provider_requests"] == output["model_loads"] == 0
    assert output["gpu_requests"] == output["network_requests"] == 0
    assert output["request2_eligible"] is False
    assert output["training_authorized"] is False
