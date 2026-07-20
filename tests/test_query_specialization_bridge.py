from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from anchor_mvp.research.query_specialization import build_training_view


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/research/materialize_query_specialization_sft.py"
CONFIG_PATH = REPO_ROOT / "configs/research/query_specialization_mvp.yaml"


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location("query_specialization_bridge", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialized_view_is_completion_only_and_keeps_lineage() -> None:
    module = _load_bridge_module()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sidecars, _, _, _ = module._load_contract_dataset(config)
    sidecar = sidecars[0]
    record = sidecar.training_record

    view = module.materialized_view(sidecar)

    assert view["schema_version"] == "anchor.query-specialization-sft-view.v1"
    assert view["source_sidecar_record_id"] == sidecar.record_id
    assert len(view["source_sidecar_record_sha256"]) == 64
    assert view["source_gold_record_id"] == sidecar.source_gold_record_id
    assert view["source_snapshot_sha256"] == sidecar.source_snapshot_sha256
    assert view["base_task_board_sha256"] == sidecar.base_task_board_sha256
    assert view["source_augmentation"]["kind"] == sidecar.augmentation.kind
    assert [message["role"] for message in view["messages"]] == [
        "user",
        "assistant",
    ]
    prompt = json.loads(view["messages"][0]["content"])
    assert prompt["role"] == record.role
    assert "segment_plan" not in prompt
    assert sidecar.segment_plan_schema_sha256 == sidecar.segment_plan["bindings"][
        "segment_plan_schema_sha256"
    ]
    assert json.loads(view["messages"][1]["content"]) == json.loads(
        record.target_output
    )


def test_all_official_sidecars_preserve_relevant_and_exclude_forbidden_content() -> None:
    """Exercise the real filtering/materialization path over all 15 fixtures."""

    module = _load_bridge_module()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sidecars, _, _, _ = module._load_contract_dataset(config)

    assert len(sidecars) == 15
    for sidecar in sidecars:
        record = sidecar.training_record
        training_view = build_training_view(record)
        materialized = module.materialized_view(sidecar)
        user_content = materialized["messages"][0]["content"]
        assert user_content == training_view.prompt
        target = json.loads(record.target_output)
        assert target["answer"] not in user_content

        prompt = json.loads(user_content)
        assert "segment_plan" not in prompt
        prompt_blocks = {block["id"]: block["content"] for block in prompt["blocks"]}
        source_blocks = {block.block_id: block.content for block in record.blocks}

        for block_id in record.targets.relevant:
            assert block_id in prompt_blocks
            assert prompt_blocks[block_id] == source_blocks[block_id]
            encoded_content = json.dumps(
                source_blocks[block_id], ensure_ascii=False
            )[1:-1]
            assert encoded_content in user_content
        for block_id in record.targets.forbidden:
            assert block_id not in prompt_blocks
            encoded_content = json.dumps(
                source_blocks[block_id], ensure_ascii=False
            )[1:-1]
            assert encoded_content not in user_content


def test_materializer_shards_without_splitting_records() -> None:
    module = _load_bridge_module()
    lines = (b"aaaa\n", b"bbbb\n", b"cc\n")

    shards = module._shard_lines(lines, max_bytes=8)

    assert shards == (b"aaaa\n", b"bbbb\ncc\n")
    assert all(len(shard) <= 8 for shard in shards)


def test_bridge_dry_run_is_content_free_and_does_not_start_training() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    manifest = payload["manifest"]
    serialized = json.dumps(manifest)

    assert payload["ok"] is True
    assert manifest["mechanical_q1_smoke_started"] is False
    assert manifest["bridge_contract_passed"] is True
    assert manifest["required_splits"] == ["train", "calibration"]
    assert manifest["missing_required_splits"] == []
    assert manifest["producer_manifest"]["counts"]["total"] == 15
    assert manifest["producer_manifest"]["heldout_content_read"] is False
    assert "calibration_as_heldout_evaluation" in manifest["non_claims"]
    assert all(file["bytes"] <= manifest["max_shard_bytes"] for file in manifest["files"])
    assert "Build the requested application" not in serialized
    assert "promoted" not in serialized


def test_execute_writes_content_addressed_train_and_calibration_views(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--execute",
            "--output-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    output_dir = Path(payload["output_dir"])
    assert output_dir.name.startswith("views-")
    assert len(output_dir.name.removeprefix("views-")) == 64
    assert (output_dir / "manifest.json").is_file()
    output_manifest = json.loads((output_dir / "manifest.json").read_text("utf-8"))
    assert output_manifest["bridge_contract_passed"] is True
    assert {entry["split"] for entry in output_manifest["files"]} == {
        "train",
        "calibration",
    }
    assert not any(".eval." in entry["path"] for entry in output_manifest["files"])

    repeated = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--execute",
            "--output-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(repeated.stdout)["output_dir"] == str(output_dir)
