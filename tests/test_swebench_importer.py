from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from anchor_mvp.swebench.cli import main as swebench_main
from anchor_mvp.swebench.importer import ImportConfig, import_metadata_cards
from anchor_mvp.swebench.partition import (
    ALLOWLIST_SCHEMA_VERSION,
    HELDOUT_REGISTRY_SCHEMA_VERSION,
    LICENSE_LEDGER_SCHEMA_VERSION,
)
from anchor_mvp.swebench.schema import (
    CHAIN_INDEX_SCHEMA_VERSION,
    CHAIN_STAGES,
    SWEBenchValidationError,
    source_fingerprint,
)


TRAIN_REVISION = "a" * 40
HELDOUT_REVISION = "b" * 40
TRAIN_DATASET = "SWE-bench/SWE-smith"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle(
    tmp_path: Path,
    *,
    dataset_id: str = TRAIN_DATASET,
) -> dict[str, Any]:
    train_rows = [
        {
            "instance_id": "smith-task-1",
            "repo": "train-org/repo-one",
            "problem_statement": "Repair parser behavior for an empty token stream.",
            "image_name": "swe-smith.python.repo-one:task-1",
            "patch": "SECRET_PATCH_ONE",
            "test_patch": "SECRET_TEST_PATCH_ONE",
            "hints_text": "SECRET_HINT_ONE",
            "FAIL_TO_PASS": ["test_secret_failure"],
            "PASS_TO_PASS": ["test_secret_regression"],
            "tests": ["SECRET_TEST_BODY"],
        },
        {
            "instance_id": "smith-task-2",
            "repo": "train-org/repo-two",
            "problem_statement": "Preserve ordering when duplicate nodes are merged.",
            "image_name": "swe-smith.python.repo-two:task-2",
            "patch": "SECRET_PATCH_TWO",
        },
        {
            "instance_id": "not-allowlisted",
            "repo": "unreviewed/repo",
            "problem_statement": "This row must be skipped before the license gate.",
            "image_name": "swe-smith.python.unreviewed:unused",
        },
    ]
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, train_rows)

    heldout_specs = [
        (
            "full",
            "SWE-bench/SWE-bench",
            "held-full-1",
            "held-org/full-repo",
            "1" * 40,
        ),
        (
            "lite",
            "SWE-bench/SWE-bench_Lite",
            "held-lite-1",
            "held-org/lite-repo",
            "2" * 40,
        ),
        (
            "verified",
            "SWE-bench/SWE-bench_Verified",
            "held-verified-1",
            "held-org/verified-repo",
            "3" * 40,
        ),
    ]
    registry_sources = []
    heldout_rows: dict[str, dict[str, Any]] = {}
    for variant, heldout_dataset, instance_id, repo, base_commit in heldout_specs:
        row = {
            "instance_id": instance_id,
            "repo": repo,
            "problem_statement": f"Permanent held-out {variant} issue.",
            "base_commit": base_commit,
        }
        heldout_rows[variant] = row
        path = tmp_path / f"heldout-{variant}.jsonl"
        _write_jsonl(path, [row])
        registry_sources.append(
            {
                "dataset_id": heldout_dataset,
                "dataset_revision": HELDOUT_REVISION,
                "split": "test",
                "metadata_jsonl": path.name,
            }
        )
    registry = tmp_path / "heldout-registry.json"
    _write_json(
        registry,
        {
            "schema_version": HELDOUT_REGISTRY_SCHEMA_VERSION,
            "sources": registry_sources,
        },
    )

    allowlist = tmp_path / "allowlist.json"
    _write_json(
        allowlist,
        {
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_revision": TRAIN_REVISION,
            "split": "train",
            "instance_ids": ["smith-task-1", "smith-task-2"],
        },
    )
    ledger = tmp_path / "license-ledger.json"
    repositories = {
        repo: {
            "spdx_id": "MIT",
            "license_file_sha256": character * 64,
            "reviewed": True,
            "training_allowed": True,
            "metadata_redistribution_allowed": True,
            "attribution": f"Fixture attribution for {repo}",
        }
        for repo, character in (
            ("train-org/repo-one", "4"),
            ("train-org/repo-two", "5"),
        )
    }
    _write_json(
        ledger,
        {
            "schema_version": LICENSE_LEDGER_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_revision": TRAIN_REVISION,
            "repositories": repositories,
        },
    )
    config = ImportConfig(
        source_jsonl=source,
        dataset_id=dataset_id,
        dataset_revision=TRAIN_REVISION,
        train_allowlist=allowlist,
        heldout_registry=registry,
        license_ledger=ledger,
        cards_output=tmp_path / "cards.jsonl",
        manifest_output=tmp_path / "manifest.json",
        dry_run=True,
    )
    return {
        "config": config,
        "train_rows": train_rows,
        "heldout_rows": heldout_rows,
        "registry_sources": registry_sources,
        "repositories": repositories,
    }


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key.casefold())
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def _chain_rows(result: Any) -> list[dict[str, Any]]:
    rows = []
    for card in result.cards:
        route = {
            "domain_id": card.domain_id,
            "builder_expert_id": card.builder_expert_id,
            "reviewer_expert_id": card.reviewer_expert_id,
        }
        rows.append(
            {
                "schema_version": CHAIN_INDEX_SCHEMA_VERSION,
                "alignment_id": card.alignment_id,
                "completed_stages": list(CHAIN_STAGES),
                "execution_sandbox_audit_sha256": "6" * 64,
                "planner_route": route,
                "executed_route": dict(route),
            }
        )
    return rows


def test_dry_run_emits_cards_without_oracle_fields_or_files(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    config = bundle["config"]
    result = import_metadata_cards(config)

    assert len(result.cards) == 2
    assert len({card.alignment_id for card in result.cards}) == 2
    assert not config.cards_output.exists()
    assert not config.manifest_output.exists()
    for card in result.cards:
        payload = card.to_dict()
        assert payload["domain_id"] == "python-repository"
        assert payload["language"] == "python"
        assert payload["task_kind"] == "issue-resolution"
        assert payload["routing_contract"] == {
            "builder_expert_id": "swe-shared-builder",
            "reviewer_expert_id": "swe-shared-reviewer",
        }
        assert not {
            "patch",
            "test_patch",
            "hints_text",
            "fail_to_pass",
            "pass_to_pass",
            "tests",
        } & _all_keys(payload)

    manifest_text = json.dumps(result.manifest, sort_keys=True)
    for forbidden in (
        "smith-task-1",
        "smith-task-2",
        "Repair parser behavior",
        "SECRET_PATCH",
        "train-org/repo-one",
    ):
        assert forbidden not in manifest_text
    assert result.manifest["content_emitted"] is False
    assert result.manifest["cards"]["card_count"] == 2
    assert result.manifest["cards"]["planned_chain_count"] == 2
    assert result.manifest["complete_chains"]["complete_chain_count"] == 0
    assert result.manifest["complete_chains"]["coverage_complete"] is False


def test_non_dry_run_writes_deterministic_cards_manifest_and_sidecar(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    config = replace(bundle["config"], dry_run=False)
    result = import_metadata_cards(config)

    cards_bytes = config.cards_output.read_bytes()
    assert (
        sha256(cards_bytes).hexdigest() == result.manifest["cards"]["cards_file_sha256"]
    )
    assert _read_json(config.manifest_output) == result.manifest
    sidecar = config.manifest_output.with_suffix(".json.sha256")
    assert sha256(config.manifest_output.read_bytes()).hexdigest() in sidecar.read_text(
        encoding="ascii"
    )

    bundle["train_rows"].reverse()
    _write_jsonl(config.source_jsonl, bundle["train_rows"])
    second = import_metadata_cards(config)
    assert (
        second.manifest["cards"]["cards_file_sha256"]
        == result.manifest["cards"]["cards_file_sha256"]
    )


def test_cli_dry_run_prints_manifest_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _bundle(tmp_path)["config"]
    code = swebench_main(
        [
            "import",
            "--source-jsonl",
            str(config.source_jsonl),
            "--dataset-id",
            config.dataset_id,
            "--dataset-revision",
            config.dataset_revision,
            "--train-allowlist",
            str(config.train_allowlist),
            "--heldout-registry",
            str(config.heldout_registry),
            "--license-ledger",
            str(config.license_ledger),
            "--cards-output",
            str(config.cards_output),
            "--manifest-output",
            str(config.manifest_output),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["content_emitted"] is False
    assert captured.err == ""
    assert not config.cards_output.exists()
    assert not config.manifest_output.exists()


def test_revision_must_be_full_immutable_commit(tmp_path: Path) -> None:
    config = _bundle(tmp_path)["config"]
    with pytest.raises(SWEBenchValidationError, match="full immutable 40-hex"):
        import_metadata_cards(replace(config, dataset_revision="main"))


def test_ordinary_swebench_train_split_is_supported(tmp_path: Path) -> None:
    config = _bundle(tmp_path, dataset_id="SWE-bench/SWE-bench")["config"]
    result = import_metadata_cards(config)
    assert len(result.cards) == 2
    assert result.manifest["source"]["split"] == "train"


@pytest.mark.parametrize(
    "dataset_id",
    ["SWE-bench/SWE-bench_Lite", "SWE-bench/SWE-bench_Verified"],
)
def test_lite_and_verified_are_never_train_sources(
    tmp_path: Path, dataset_id: str
) -> None:
    config = _bundle(tmp_path, dataset_id=dataset_id)["config"]
    with pytest.raises(SWEBenchValidationError, match="permanent held-out"):
        import_metadata_cards(config)


def test_non_train_split_is_rejected(tmp_path: Path) -> None:
    config = _bundle(tmp_path)["config"]
    with pytest.raises(SWEBenchValidationError, match="split=train"):
        import_metadata_cards(replace(config, split="dev"))


def test_registry_must_cover_full_lite_and_verified(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    config = bundle["config"]
    _write_json(
        config.heldout_registry,
        {
            "schema_version": HELDOUT_REGISTRY_SCHEMA_VERSION,
            "sources": bundle["registry_sources"][:-1],
        },
    )
    with pytest.raises(SWEBenchValidationError, match="Full, Lite, and Verified"):
        import_metadata_cards(config)


def test_exact_heldout_id_overlap_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    config = bundle["config"]
    allowlist = _read_json(config.train_allowlist)
    allowlist["instance_ids"][0] = "held-full-1"
    _write_json(config.train_allowlist, allowlist)
    with pytest.raises(SWEBenchValidationError, match="intersects"):
        import_metadata_cards(config)


def test_heldout_fingerprint_and_repository_overlap_are_rejected(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    config = bundle["config"]
    heldout = bundle["heldout_rows"]["full"]
    row = bundle["train_rows"][0]
    row.update(
        repo=heldout["repo"],
        problem_statement=heldout["problem_statement"],
        base_commit=heldout["base_commit"],
    )
    row.pop("image_name")
    _write_jsonl(config.source_jsonl, bundle["train_rows"])
    with pytest.raises(SWEBenchValidationError, match="source fingerprint"):
        import_metadata_cards(config)

    row["problem_statement"] = "Different issue text in the same held-out repository."
    _write_jsonl(config.source_jsonl, bundle["train_rows"])
    with pytest.raises(SWEBenchValidationError, match="repository denylist"):
        import_metadata_cards(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewed", False),
        ("training_allowed", False),
        ("metadata_redistribution_allowed", False),
    ],
)
def test_license_ledger_is_fail_closed(tmp_path: Path, field: str, value: bool) -> None:
    bundle = _bundle(tmp_path)
    config = bundle["config"]
    ledger = _read_json(config.license_ledger)
    ledger["repositories"]["train-org/repo-one"][field] = value
    _write_json(config.license_ledger, ledger)
    with pytest.raises(SWEBenchValidationError, match="explicitly approved"):
        import_metadata_cards(config)


def test_missing_repository_license_and_missing_source_id_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    config = bundle["config"]
    ledger = _read_json(config.license_ledger)
    del ledger["repositories"]["train-org/repo-one"]
    _write_json(config.license_ledger, ledger)
    with pytest.raises(SWEBenchValidationError, match="reviewed license ledger"):
        import_metadata_cards(config)

    _write_json(
        config.license_ledger,
        {
            "schema_version": LICENSE_LEDGER_SCHEMA_VERSION,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "repositories": bundle["repositories"],
        },
    )
    allowlist = _read_json(config.train_allowlist)
    allowlist["instance_ids"].append("missing-task")
    _write_json(config.train_allowlist, allowlist)
    with pytest.raises(SWEBenchValidationError, match="absent from source"):
        import_metadata_cards(config)


def test_complete_chain_count_requires_exact_route_and_stage_coverage(
    tmp_path: Path,
) -> None:
    config = _bundle(tmp_path)["config"]
    initial = import_metadata_cards(config)
    chain_index = tmp_path / "chain-index.jsonl"
    rows = _chain_rows(initial)
    _write_jsonl(chain_index, rows)

    complete = import_metadata_cards(replace(config, chain_index=chain_index))
    assert complete.manifest["complete_chains"]["complete_chain_count"] == 2
    assert complete.manifest["complete_chains"]["coverage_complete"] is True

    _write_jsonl(chain_index, rows[:-1])
    with pytest.raises(SWEBenchValidationError, match="count must equal"):
        import_metadata_cards(replace(config, chain_index=chain_index))

    bad_stages = _chain_rows(initial)
    bad_stages[0]["completed_stages"].pop()
    _write_jsonl(chain_index, bad_stages)
    with pytest.raises(SWEBenchValidationError, match="ordered stage contract"):
        import_metadata_cards(replace(config, chain_index=chain_index))

    bad_route = _chain_rows(initial)
    bad_route[0]["executed_route"]["builder_expert_id"] = "wrong-builder"
    _write_jsonl(chain_index, bad_route)
    with pytest.raises(SWEBenchValidationError, match="executed experts"):
        import_metadata_cards(replace(config, chain_index=chain_index))


def test_source_fingerprint_is_stable_and_binds_issue_and_workspace() -> None:
    values = {
        "repo": "owner/repository",
        "problem_statement": "Fix  repeated   whitespace.",
        "base_commit": "7" * 40,
    }
    first = source_fingerprint(**values)
    assert first == source_fingerprint(
        **{**values, "problem_statement": " fix repeated whitespace. "}
    )
    assert first != source_fingerprint(
        **{**values, "problem_statement": "Fix another issue."}
    )
    assert first != source_fingerprint(**{**values, "base_commit": "8" * 40})
