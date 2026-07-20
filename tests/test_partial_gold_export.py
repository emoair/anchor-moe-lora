from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import yaml

from anchor_mvp.data.automation import (
    AutomationConfig,
    AutomationRunner,
    main,
    partition_collected_records,
)
from anchor_mvp.data.partial_export import (
    EXPORT_SCHEMA_VERSION,
    TRAINING_MODE,
    export_partial_expert_gold,
)
from anchor_mvp.data.pipeline import DistillationPipeline
from anchor_mvp.data.schema import TASK_TYPES
from anchor_mvp.data.teacher import MockTeacher


ROOT = Path(__file__).resolve().parents[1]


def _partial_fixture(tmp_path: Path) -> tuple[AutomationConfig, dict]:
    settings = AutomationConfig(
        sop_dir=ROOT / "skills",
        output_dir=tmp_path / "collection",
        concurrency_stages=(1,),
        stage_seed_counts=(2,),
        collection_policy="collect_then_partition",
        minimum_label_counts={"security": {"BLOCK": 2}},
    )
    report = asyncio.run(
        DistillationPipeline(
            teacher=MockTeacher(),
            sop_dir=settings.sop_dir,
            output_dir=settings.output_dir,
            task_card_config=settings.task_card_config,
        ).run(seed_count=2)
    )
    assert report.errors == ()
    partition = partition_collected_records(settings, 2)
    assert partition["training_ready"] is False
    runner = AutomationRunner(config=settings, teacher=MockTeacher())
    runner.status.update(
        {
            "state": "gate_blocked",
            "current_worker": None,
            "current_concurrency": 0,
            "partition": partition,
        }
    )
    runner._save_status()
    return settings, partition


def _destination(settings: AutomationConfig) -> Path:
    partition_sha = hashlib.sha256(
        (settings.partition_dir / "manifest.json").read_bytes()
    ).hexdigest()
    return settings.output_dir / "training_exports" / TRAINING_MODE / partition_sha


def test_partial_export_copies_only_bound_strict_gold_with_explicit_waivers(
    tmp_path: Path,
) -> None:
    settings, partition = _partial_fixture(tmp_path)

    manifest = export_partial_expert_gold(settings)
    destination = _destination(settings)

    assert manifest["schema_version"] == EXPORT_SCHEMA_VERSION
    assert manifest["training_mode"] == "per_expert_partial_gold"
    assert manifest["not_for_end_to_end_claim"] is True
    assert manifest["strict_complete_chains"] == partition["complete_chain_count"]
    assert manifest["waivers"]["label_quota"]["applied"] is True
    assert manifest["excluded"] == {
        "negative": True,
        "reject": True,
        "oracle_label_only": True,
        "heldout": True,
    }
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        *(f"data_{task}.jsonl" for task in TASK_TYPES),
    }
    for task in TASK_TYPES:
        source = settings.partition_dir / "gold" / f"data_{task}.jsonl"
        output = destination / f"data_{task}.jsonl"
        assert output.read_bytes() == source.read_bytes()
        assert (
            manifest["gold_files"][task]["sha256"]
            == hashlib.sha256(output.read_bytes()).hexdigest()
        )
    assert not (destination / "negative.jsonl").exists()
    assert not (destination / "reject.jsonl").exists()
    assert export_partial_expert_gold(settings) == manifest


def test_partial_export_fails_closed_for_active_or_hash_stale_partition(
    tmp_path: Path,
) -> None:
    settings, _partition = _partial_fixture(tmp_path)
    status = json.loads(settings.status_path.read_text(encoding="utf-8"))
    status["state"] = "running"
    status["current_worker"] = "review"
    settings.status_path.write_text(json.dumps(status), encoding="utf-8")
    try:
        export_partial_expert_gold(settings)
    except ValueError as error:
        assert "stopped" in str(error)
    else:
        raise AssertionError("active automation must not export")

    status["state"] = "gate_blocked"
    status["current_worker"] = None
    settings.status_path.write_text(json.dumps(status), encoding="utf-8")
    with (settings.partition_dir / "negative.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write('{"unexpected":"negative"}\n')
    try:
        export_partial_expert_gold(settings)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("stale partition must not export")


def test_partial_export_cli_is_explicit_offline_and_emits_metadata_only(
    tmp_path: Path, capsys
) -> None:
    settings, _partition = _partial_fixture(tmp_path)
    config_path = tmp_path / "automation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sop_dir": str(settings.sop_dir),
                "output_dir": str(settings.output_dir),
                "concurrency_stages": [1],
                "stage_seed_counts": [2],
                "collection_policy": "collect_then_partition",
                "minimum_label_counts": {"security": {"BLOCK": 2}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--export-partial-gold"]) == 0
    output = capsys.readouterr().out
    assert '"training_mode": "per_expert_partial_gold"' in output
    assert '"messages"' not in output
    assert '"input"' not in output


def test_partial_export_rejects_heldout_provenance_even_when_metadata_is_rebound(
    tmp_path: Path,
) -> None:
    settings, partition = _partial_fixture(tmp_path)
    plan_path = settings.partition_dir / "gold" / "data_plan.jsonl"
    rows = [
        json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["provenance"]["source_kind"] = "swebench_heldout"
    plan_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    partition["gold_files"]["plan"].update(
        {
            "records": len(rows),
            "bytes": plan_path.stat().st_size,
            "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        }
    )
    (settings.partition_dir / "manifest.json").write_text(
        json.dumps(partition), encoding="utf-8"
    )
    status = json.loads(settings.status_path.read_text(encoding="utf-8"))
    status["partition"] = partition
    settings.status_path.write_text(json.dumps(status), encoding="utf-8")

    try:
        export_partial_expert_gold(settings)
    except ValueError as error:
        assert "held-out" in str(error)
    else:
        raise AssertionError("held-out rows must never be exported")
