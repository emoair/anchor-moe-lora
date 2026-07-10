from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.config import load_training_config  # noqa: E402
from anchor_mvp.training.preflight import (  # noqa: E402
    build_preflight_report,
    verify_prior_smoke_gate,
)


CONFIG = ROOT / "configs" / "training" / "gemma4_12b_qlora_smoke.yaml"


def canonical_record(expert: str, identifier: str, *, live: bool = True) -> dict:
    if expert == "security_audit":
        assistant = "[BLOCK]"
        output = {"decision": "BLOCK", "rationale": "Untrusted HTML reaches a DOM sink."}
    else:
        assistant = "export const value = 1;"
        output = {"code": assistant}
    teacher = {
        "model": "live-teacher" if live else "mock-teacher",
        "base_url": "https://teacher.example/v1" if live else "mock://teacher",
    }
    return {
        "schema_version": "1.0",
        "id": identifier,
        "expert": expert,
        "messages": [
            {"role": "user", "content": "Do the bounded task."},
            {"role": "assistant", "content": assistant},
        ],
        "decision_trace": [
            {"check": "contract", "evidence": "fixture", "action": "return target"}
        ],
        "output": output,
        "provenance": {"teacher": teacher},
    }


def ready_dependencies() -> dict:
    return {
        "ready": True,
        "missing": [],
        "incompatible": [],
        "device": {
            "cuda_available": True,
            "bf16_supported": True,
            "free_memory_gib": 11.5,
            "name": "Fake GPU",
        },
        "host_memory": {
            "probed": True,
            "available_memory_gib": 16.0,
            "total_memory_gib": 24.0,
        },
    }


def fixture_config(tmp_path: Path, *, omit: str | None = None, live: bool = True) -> dict:
    config = copy.deepcopy(load_training_config(CONFIG))
    config["paths"]["project_root"] = "."
    mapping = {
        "frontend_gen": "data/live_smoke/data_frontend.jsonl",
        "code_review": "data/live_smoke/data_review.jsonl",
        "security_audit": "data/live_smoke/data_security.jsonl",
    }
    config["scale_gate"]["required_datasets"] = mapping
    for expert, relative in mapping.items():
        if expert == omit:
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(canonical_record(expert, f"{expert}-1", live=live)) + "\n",
            encoding="utf-8",
        )

    heldout = tmp_path / "configs/training/heldout_cases.jsonl"
    heldout.parent.mkdir(parents=True, exist_ok=True)
    heldout.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"heldout-{expert}",
                    "expert": expert,
                    "prompt": "Probe this expert.",
                    "max_new_tokens": 4,
                }
            )
            for expert in mapping
        )
        + "\n",
        encoding="utf-8",
    )
    config["scale_gate"]["heldout_cases"] = "configs/training/heldout_cases.jsonl"

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"tiny-weight-fixture")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    download_manifest = {
        "repo_id": config["model"]["id"],
        "revision": config["model"]["revision"],
        "verification": {
            "file": weight.name,
            "bytes": weight.stat().st_size,
            "sha256": digest,
            "matches_hugging_face_lfs_oid": True,
        },
    }
    (model_dir / "download.json").write_text(
        json.dumps(download_manifest), encoding="utf-8"
    )
    config["scale_gate"]["base_artifact"] = {
        "repo_id": config["model"]["id"],
        "revision": config["model"]["revision"],
        "local_path": "model",
        "download_manifest": "download.json",
        "weight_file": weight.name,
        "bytes": weight.stat().st_size,
        "sha256": digest,
    }
    config["scale_gate"]["minimum_free_vram_gib"] = 10.5
    config["scale_gate"]["minimum_free_host_memory_gib"] = 12.0
    config["scale_gate"]["required_smoke_gate_manifest"] = "artifacts/smoke.json"
    return config


def test_preflight_passes_complete_live_canonical_fixture(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    report, cases = build_preflight_report(config, tmp_path, ready_dependencies())
    assert report["passed"] is True
    assert len(cases) == 3
    assert all(gate["passed"] for gate in report["gates"].values())
    assert report["base"]["checksum_source"] == "verified-download-manifest"


def test_preflight_blocks_when_any_expert_file_is_missing(tmp_path: Path) -> None:
    config = fixture_config(tmp_path, omit="security_audit")
    report, _ = build_preflight_report(config, tmp_path, ready_dependencies())
    assert report["passed"] is False
    assert report["gates"]["three_live_datasets_present"]["passed"] is False
    assert report["gates"]["canonical_schema_valid"]["passed"] is False


def test_preflight_rejects_mock_teacher_records(tmp_path: Path) -> None:
    config = fixture_config(tmp_path, live=False)
    report, _ = build_preflight_report(config, tmp_path, ready_dependencies())
    assert report["passed"] is False
    assert report["gates"]["real_teacher_samples"]["passed"] is False


def test_preflight_blocks_when_host_memory_headroom_is_too_low(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    dependencies = ready_dependencies()
    dependencies["host_memory"]["available_memory_gib"] = 5.17
    report, _ = build_preflight_report(config, tmp_path, dependencies)
    gate = report["gates"]["host_free_memory"]
    assert report["passed"] is False
    assert gate["passed"] is False
    assert gate["evidence"]["free_gib"] == 5.17
    assert gate["evidence"]["required_gib"] == 12.0


def test_prior_smoke_manifest_must_match_dataset_snapshot(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    report, _ = build_preflight_report(config, tmp_path, ready_dependencies())
    manifest_path = tmp_path / config["scale_gate"]["required_smoke_gate_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "smoke-gate",
                "mode": "execute",
                "smoke_gate": {"passed": True},
                "base_model_revision": config["model"]["revision"],
                "preflight": {
                    "dataset_snapshot_sha256": report["dataset_snapshot_sha256"]
                },
            }
        ),
        encoding="utf-8",
    )
    assert verify_prior_smoke_gate(config, tmp_path, report)["passed"] is True
    changed = dict(report)
    changed["dataset_snapshot_sha256"] = "different"
    assert verify_prior_smoke_gate(config, tmp_path, changed)["passed"] is False
