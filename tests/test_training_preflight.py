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
    inspect_training_artifact,
    verify_prior_smoke_gate,
)


CONFIG = ROOT / "configs" / "training" / "gemma4_12b_qlora_smoke.yaml"


def canonical_record(expert: str, identifier: str, *, live: bool = True) -> dict:
    if expert == "security_gate":
        assistant = "[BLOCK]"
        output = {"decision": "BLOCK", "rationale": "Untrusted HTML reaches a DOM sink."}
    elif expert == "tool_policy":
        assistant = "APPROVE"
        output = {"decision": "APPROVE", "rationale": "Only inert local labels are proposed."}
    elif expert == "planner":
        output = {
            "summary": "Produce one bounded component.",
            "steps": [{"id": "p1", "goal": "Implement", "deliverable": "Component"}],
        }
        assistant = json.dumps(output)
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
        "planner": "data/live_smoke/data_plan.jsonl",
        "tool_policy": "data/live_smoke/data_tool_policy.jsonl",
        "frontend_gen": "data/live_smoke/data_frontend.jsonl",
        "frontend_review": "data/live_smoke/data_review.jsonl",
        "security_gate": "data/live_smoke/data_security.jsonl",
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
    nf4_dir = tmp_path / "model-nf4"
    nf4_dir.mkdir()
    shard = nf4_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"tiny-nf4-shard")
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    nf4_manifest = {
        "schema_version": "anchor.bnb-nf4-export.v1",
        "model_footprint_bytes": shard.stat().st_size,
        "source": "model",
        "source_weight_sha256": digest,
        "quantization": {
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "storage_dtype": "bfloat16",
        },
        "weights": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": shard_sha,
            }
        ],
    }
    (nf4_dir / "anchor_quantization_manifest.json").write_text(
        json.dumps(nf4_manifest), encoding="utf-8"
    )
    (nf4_dir / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "load_in_8bit": False,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_quant_storage": "bfloat16",
                    "llm_int8_enable_fp32_cpu_offload": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (nf4_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {
                    "layer.weight": shard.name,
                    "layer.weight.quant_state.bitsandbytes__nf4": shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    config["model"]["local_path"] = "model-nf4"
    config["scale_gate"]["training_artifact"] = {
        "format": "transformers-bitsandbytes-nf4",
        "local_path": "model-nf4",
        "manifest": "model-nf4/anchor_quantization_manifest.json",
        "model_footprint_bytes": shard.stat().st_size,
    }
    config["scale_gate"]["minimum_free_vram_gib"] = 10.5
    config["scale_gate"]["minimum_free_host_memory_gib"] = 12.0
    config["scale_gate"]["required_smoke_gate_manifest"] = "artifacts/smoke.json"
    return config


def test_preflight_passes_complete_live_canonical_fixture(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    report, cases = build_preflight_report(config, tmp_path, ready_dependencies())
    assert report["passed"] is True
    assert len(cases) == 5
    assert all(gate["passed"] for gate in report["gates"].values())
    assert report["base"]["checksum_source"] == "verified-download-manifest"


def test_preflight_blocks_when_any_expert_file_is_missing(tmp_path: Path) -> None:
    config = fixture_config(tmp_path, omit="security_gate")
    report, _ = build_preflight_report(config, tmp_path, ready_dependencies())
    assert report["passed"] is False
    assert report["gates"]["five_live_datasets_present"]["passed"] is False
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


def test_training_artifact_gate_binds_reloadable_nf4_shards(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    report = inspect_training_artifact(config, tmp_path)
    assert report["passed"] is True
    assert report["checksum_source"] == "manifest-and-file-size"
    assert report["checks"]["frozen_peft_contract"] is True
    assert report["index_checks"] == {
        "shards_exact": True,
        "total_size_plausible": True,
        "nf4_quant_state": True,
    }


def test_training_artifact_gate_rejects_wrong_quant_type_or_shard_size(
    tmp_path: Path,
) -> None:
    config = fixture_config(tmp_path)
    nf4_dir = tmp_path / "model-nf4"
    model_config = json.loads((nf4_dir / "config.json").read_text(encoding="utf-8"))
    model_config["quantization_config"]["bnb_4bit_quant_type"] = "fp4"
    (nf4_dir / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (nf4_dir / "model-00001-of-00001.safetensors").write_bytes(b"drift")

    report = inspect_training_artifact(config, tmp_path)
    assert report["passed"] is False
    assert "NF4 training artifact contract failed: transformers_config" in report[
        "errors"
    ]
    assert "NF4 weight binding failed at index 0" in report["errors"]
