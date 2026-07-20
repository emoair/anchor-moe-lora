from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.data.partial_export import EXPORT_SCHEMA_VERSION, TRAINING_MODE
from anchor_mvp.data.partial_snapshot import (
    DEFAULT_RECORDS_PER_EXPERT,
    EXPERT_SOURCES,
    PARTIAL_SNAPSHOT_SCHEMA,
    freeze_partial_gold_snapshot,
    main,
)
from anchor_mvp.training.config import (
    ConfigError,
    load_training_config,
    validate_training_config,
)
from anchor_mvp.training.preflight import (
    inspect_dataset_snapshot_manifest,
    inspect_gate_datasets,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/formal_partial_v1_lowmem_common.yaml"
SMOKE = ROOT / "configs/training/formal_partial_v1_lowmem_smoke.yaml"


def _canonical_record(expert: str, identifier: str) -> dict:
    if expert == "planner":
        output = {
            "summary": "Produce one bounded component.",
            "steps": [{"id": "p1", "goal": "Implement", "deliverable": "Code"}],
        }
        assistant = json.dumps(output, ensure_ascii=False, sort_keys=True)
    elif expert == "tool_policy":
        output = {"decision": "APPROVE", "rationale": "Only inert local edits."}
        assistant = "APPROVE"
    elif expert == "security_gate":
        output = {"decision": "PASS", "rationale": "No unsafe sink is present."}
        assistant = "[PASS]"
    else:
        output = {"code": f"export const value = {identifier!r};"}
        assistant = output["code"]
    return {
        "schema_version": "1.0",
        "id": identifier,
        "expert": expert,
        "messages": [
            {"role": "user", "content": "Perform the bounded task."},
            {"role": "assistant", "content": assistant},
        ],
        "decision_trace": [
            {"check": "contract", "evidence": "fixture", "action": "return"}
        ],
        "output": output,
        "provenance": {
            "teacher": {
                "model": "live-teacher",
                "base_url": "https://teacher.example/v1",
            },
            "source_kind": "self_synthetic",
        },
    }


def _export_fixture(tmp_path: Path, *, rows: int = 129) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    gold_files: dict[str, dict] = {}
    gold_records: dict[str, int] = {}
    for expert, (task, filename) in EXPERT_SOURCES.items():
        path = export_dir / filename
        path.write_text(
            "".join(
                json.dumps(_canonical_record(expert, f"{expert}-{index}")) + "\n"
                for index in range(rows)
            ),
            encoding="utf-8",
        )
        gold_files[task] = {
            "path": filename,
            "records": rows,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        gold_records[task] = rows
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "training_mode": TRAINING_MODE,
        "not_for_end_to_end_claim": True,
        "source": {"partition_manifest_sha256": "a" * 64},
        "strict_complete_chains": 80,
        "gold_files": gold_files,
        "gold_records_by_task": gold_records,
        "waivers": {"complete_chain": {"applied": True}},
        "excluded": {
            "negative": True,
            "reject": True,
            "oracle_label_only": True,
            "heldout": True,
        },
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return export_dir


def test_partial_snapshot_freezes_balanced_128_and_preflight_accepts_it(
    tmp_path: Path,
) -> None:
    export_dir = _export_fixture(tmp_path)
    output_dir = tmp_path / "artifacts/formal_partial_v1/dataset"

    manifest = freeze_partial_gold_snapshot(export_dir, output_dir)

    assert manifest["schema_version"] == PARTIAL_SNAPSHOT_SCHEMA
    assert manifest["training_mode"] == TRAINING_MODE
    assert manifest["not_for_end_to_end_claim"] is True
    assert manifest["per_expert"] == DEFAULT_RECORDS_PER_EXPERT
    assert manifest["total_records"] == 5 * DEFAULT_RECORDS_PER_EXPERT
    assert all(
        item["records"] == DEFAULT_RECORDS_PER_EXPERT
        and item["source_records"] == 129
        for item in manifest["files"].values()
    )
    assert manifest["excluded"] == {
        "negative": True,
        "oracle_label_only": True,
        "reject": True,
        "heldout": True,
    }
    assert freeze_partial_gold_snapshot(export_dir, output_dir) == manifest

    config = load_training_config(CONFIG)
    datasets = inspect_gate_datasets(config, tmp_path)
    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)
    assert datasets["schemas_valid"] is True
    assert datasets["all_records_live"] is True
    assert report["passed"] is True
    assert report["not_for_end_to_end_claim"] is True
    assert all(
        all(checks.values()) for checks in report["file_checks"].values()
    )


def test_partial_snapshot_fails_closed_if_an_exclusion_proof_is_removed(
    tmp_path: Path,
) -> None:
    export_dir = _export_fixture(tmp_path)
    source_manifest = export_dir / "manifest.json"
    value = json.loads(source_manifest.read_text(encoding="utf-8"))
    value["excluded"]["heldout"] = False
    source_manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="contract is incomplete"):
        freeze_partial_gold_snapshot(export_dir, tmp_path / "snapshot")


def test_partial_snapshot_config_requires_explicit_scope_and_equal_128() -> None:
    config = load_training_config(CONFIG)
    smoke = load_training_config(SMOKE)
    assert config["scale_gate"]["dataset_snapshot"] == {
        "schema_version": PARTIAL_SNAPSHOT_SCHEMA,
        "manifest": "artifacts/formal_partial_v1/dataset/manifest.json",
        "sidecar": "artifacts/formal_partial_v1/dataset/manifest.json.sha256",
        "immutable": True,
        "minimum_records_per_expert": 128,
        "training_mode": TRAINING_MODE,
        "not_for_end_to_end_claim": True,
        "balanced_records_per_expert": 128,
    }
    assert smoke["training"]["max_steps"] == 1
    assert smoke["training"]["gradient_accumulation_steps"] == 1

    invalid = copy.deepcopy(config)
    invalid["scale_gate"]["dataset_snapshot"]["not_for_end_to_end_claim"] = False
    with pytest.raises(ConfigError, match="not_for_end_to_end_claim"):
        validate_training_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["scale_gate"]["dataset_snapshot"]["minimum_records_per_expert"] = 127
    with pytest.raises(ConfigError, match="balanced 128/expert"):
        validate_training_config(invalid)


def test_partial_snapshot_cli_emits_metadata_only(tmp_path: Path, capsys) -> None:
    export_dir = _export_fixture(tmp_path, rows=128)
    output_dir = tmp_path / "snapshot"

    assert (
        main(
            [
                "--export-dir",
                str(export_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"per_expert": 128' in output
    assert '"not_for_end_to_end_claim": true' in output
    assert '"messages"' not in output
    assert '"output"' not in output
