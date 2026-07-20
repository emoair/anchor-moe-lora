from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.config import (  # noqa: E402
    ConfigError,
    load_training_config,
    select_adapter,
    validate_training_config,
)
from anchor_mvp.training.preflight import (  # noqa: E402
    REQUIRED_EXPERTS,
    inspect_dataset_snapshot_manifest,
)
from anchor_mvp.training.formal_v3_schedule import (  # noqa: E402
    CANDIDATE_TASKS_PER_STAGE,
    CANDIDATE_WORK_ORDERS,
    derive_exposure_plan,
)
from anchor_mvp.training.manifest import build_manifest  # noqa: E402


CONFIG_DIR = ROOT / "configs" / "training"
COMMON = CONFIG_DIR / "formal_v3_lowmem_common.yaml"
PROFILES = {
    "A": CONFIG_DIR / "formal_v3_lowmem_base.yaml",
    "B": CONFIG_DIR / "formal_v3_lowmem_mixed.yaml",
    "C": COMMON,
    "D": CONFIG_DIR / "formal_v3_lowmem_budget.yaml",
    "E": CONFIG_DIR / "formal_v3_lowmem_adaptive.yaml",
    "F": CONFIG_DIR / "formal_v3_lowmem_adaptive_budget.yaml",
}
FILE_NAMES = {
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
FORMAL_MINIMUM = 256


def test_formal_v3_profiles_bind_only_future_immutable_full_v3_snapshot() -> None:
    for arm, path in PROFILES.items():
        profile = load_training_config(path)
        serialized = json.dumps(profile, sort_keys=True)
        assert "automated_v2" not in serialized
        assert "formal_v1/dataset" not in serialized
        assert "data/automated_v3" not in serialized
        assert profile["experiment"].startswith("anchor-moe-lora-formal-v3")
        assert profile["model"]["load_strategy"] == "prequantized_peft_4bit"
        assert profile["quantization"]["load_in_4bit"] is True
        assert profile["quantization"]["quant_type"] == "nf4"
        assert profile["quantization"]["freeze_base_model"] is True
        assert profile["training"]["runtime_engine"] == "manual_active_labels_v2"
        assert profile["training"]["max_seq_length"] == 64
        assert (
            profile["training"]["sequence_contract"]
            == "formal_v3_lowmem_truncated_v1"
        )
        assert profile["training"]["full_trajectory_training"] is False
        assert (
            profile["training"]["sample_order"]
            == "deterministic_stage_stratified_epoch_v1"
        )
        assert profile["training"]["optim"] == "paged_adamw_8bit"
        assert profile["training"]["learning_rate"] == 0.00005
        assert profile["training"]["allow_tf32"] is True
        assert profile["training"]["maximum_training_peak_vram_gib"] == 9.0
        assert profile["paths"]["adapter_dir"] == f"artifacts/formal_v3/{arm}/adapters"
        snapshot = profile["scale_gate"]["dataset_snapshot"]
        assert snapshot["schema_version"] == "anchor.training-snapshot.v2"
        assert snapshot["manifest"] == "artifacts/formal_v3/dataset/manifest.json"
        assert snapshot["sidecar"].endswith("manifest.json.sha256")
        assert snapshot["immutable"] is True
        assert snapshot["minimum_records_per_expert"] == FORMAL_MINIMUM
        fullscale = profile["scale_gate"]["formal_v3_fullscale"]
        assert fullscale["maximum_candidate_records_per_expert"] == 19_008
        assert fullscale["candidate_work_orders"] == 95_040
        assert fullscale["selection"] == "all_gold_accepted_then_split"
        assert fullscale["split_contract"] == {
            "schema_version": "anchor.formal-v3-gold-splits.v1",
            "train_role": "training_only",
            "calibration_role": "rank_allocation_only",
            "heldout_role": "evaluation_only_hash_metadata",
            "require_pairwise_disjoint": True,
            "require_heldout_content_read_false": True,
        }
        for paths in (
            profile["scale_gate"]["required_datasets"].values(),
            *(entry["datasets"] for entry in profile["adapters"].values()),
        ):
            assert all(
                value.startswith("artifacts/formal_v3/dataset/") for value in paths
            )


def test_formal_v3_exposure_budget_is_snapshot_sized_and_equal_B_through_F() -> None:
    counts = {expert: 17_105 for expert in FILE_NAMES}
    plans = {
        arm: derive_exposure_plan(
            arm=arm,
            train_records_per_expert=counts,
            gradient_accumulation_steps=4,
        )
        for arm in ("B", "C", "D", "E", "F")
    }
    assert {
        plan["arm_total_sample_exposures"] for plan in plans.values()
    } == {85_540}
    assert plans["B"]["max_steps_per_adapter_job"] == 21_385
    assert plans["C"]["max_steps_per_adapter_job"] == 4_277
    assert plans["C"]["padding_exposures_per_stage"] == 3
    assert set(plans["B"]["planned_exposures_by_stage"].values()) == {17_108}
    assert set(plans["B"]["padding_exposures_by_stage"].values()) == {3}
    assert plans["C"]["records_per_stage"] == 17_105
    assert plans["C"]["arm_total_sample_exposures"] > 640


def test_formal_v3_requires_immutable_snapshot_contract() -> None:
    profile = load_training_config(COMMON)
    invalid = copy.deepcopy(profile)
    del invalid["scale_gate"]["dataset_snapshot"]
    with pytest.raises(ConfigError, match="formal-v3 requires"):
        validate_training_config(invalid)


def test_formal_v3_rejects_legacy_128_record_contract() -> None:
    profile = load_training_config(COMMON)
    legacy = copy.deepcopy(profile)
    legacy["scale_gate"]["dataset_snapshot"]["minimum_records_per_expert"] = 128

    with pytest.raises(ConfigError, match="at least 256 records per expert"):
        validate_training_config(legacy)


def test_formal_v3_adapter_run_refuses_unmaterialized_one_step_placeholder() -> None:
    with pytest.raises(ConfigError, match="snapshot-materialized"):
        select_adapter(load_training_config(PROFILES["C"]), "planner", 16)


@pytest.mark.parametrize("rank", [1, 6, 8, 12, 16])
def test_adaptive_rank_menu_is_trainable(rank: int) -> None:
    profile = load_training_config(PROFILES["E"])
    plan = derive_exposure_plan(
        arm="E",
        train_records_per_expert={expert: 17_105 for expert in FILE_NAMES},
        gradient_accumulation_steps=4,
    )
    profile["training"]["max_steps"] = plan["max_steps_per_adapter_job"]
    profile["training"]["resolved_exposure"] = {
        **plan,
        "dataset_snapshot_sha256": "a" * 64,
        "heldout_content_read": False,
    }
    selected = select_adapter(profile, "planner", rank)
    assert selected["lora"]["rank"] == rank
    assert selected["lora"]["alpha"] == 2 * rank


def test_formal_v3_manifest_exposes_padding_and_control_invariant() -> None:
    profile = load_training_config(PROFILES["C"])
    plan = derive_exposure_plan(
        arm="C",
        train_records_per_expert={expert: 17_105 for expert in FILE_NAMES},
        gradient_accumulation_steps=4,
    )
    profile["training"]["max_steps"] = plan["max_steps_per_adapter_job"]
    profile["training"]["resolved_exposure"] = {
        **plan,
        "dataset_snapshot_sha256": "a" * 64,
        "snapshot_manifest_sha256": "b" * 64,
        "heldout_content_read": False,
    }
    selected = select_adapter(profile, "planner", 16)

    manifest = build_manifest(
        selected,
        dependency_report={"ready": True},
        datasets=[{"exists": True, "valid_records": 17_105}],
        mode="dry-run",
    )

    exposure = manifest["sample_exposure_plan"]
    assert exposure["dataset_records"] == 17_105
    assert exposure["sample_exposures"] == 17_108
    assert exposure["padding_exposures_per_stage"] == 3
    assert exposure["arm_total_sample_exposures"] == 85_540
    assert exposure["control_invariant"] == (
        "equal_total_and_per_stage_sample_exposure_B_through_F"
    )
    assert set(exposure["planned_exposures_by_stage"].values()) == {17_108}
    assert manifest["training_profile"]["full_trajectory_training"] is False
    assert manifest["training_profile"]["runtime_sequence_statistics"].startswith(
        "written_after_execution"
    )
    assert exposure["heldout_content_read"] is False


def test_schedule_materializer_writes_snapshot_bound_runnable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts/train/materialize_formal_v3_schedule.py"
    spec = importlib.util.spec_from_file_location("formal_v3_materializer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = {expert: 17_105 for expert in FILE_NAMES}
    monkeypatch.setattr(module, "inspect_gate_datasets", lambda *_: {"reports": {}})
    monkeypatch.setattr(
        module,
        "inspect_dataset_snapshot_manifest",
        lambda *_: {
            "passed": True,
            "computed_snapshot_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "split_contract": {
                "train_records_per_expert": counts,
                "heldout_ids_sha256": "c" * 64,
                "heldout_manifest_sha256": "d" * 64,
                "leakage_audit_sha256": "e" * 64,
            },
        },
    )
    output = tmp_path / "snapshot" / "C.json"

    result = module.materialize(PROFILES["C"], "C", output)
    generated = load_training_config(output)
    selected = select_adapter(generated, "planner", 16)

    assert result["training_started"] is False
    assert result["heldout_content_read"] is False
    assert Path(str(output) + ".sha256").is_file()
    assert selected["training"]["max_steps"] == 4_277
    assert selected["training"]["save_steps"] == 1_070
    assert selected["training"]["resolved_exposure"]["arm"] == "C"
    evaluation = result["evaluation_contract"]
    assert evaluation["normalization"] == "A_equals_100"
    assert evaluation["arm_topology"]["B"] == "single_mixed_adapter"
    assert evaluation["arm_topology"]["C"] == "five_stage_serial_hot_swap"
    assert evaluation["formal_v2_artifacts_allowed"] is False
    assert evaluation["evaluation_executed"] is False


def _snapshot_fixture(
    tmp_path: Path, *, records_per_expert: int = FORMAL_MINIMUM
) -> tuple[dict, dict]:
    dataset_dir = tmp_path / "artifacts" / "formal_v3" / "dataset"
    dataset_dir.mkdir(parents=True)
    reports: dict[str, dict] = {}
    manifest_files: dict[str, dict] = {}
    digest_parts: list[str] = []
    for expert in REQUIRED_EXPERTS:
        name = FILE_NAMES[expert]
        path = dataset_dir / name
        path.write_text(f"fixture:{expert}\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        reports[expert] = {
            "path": str(path.resolve()),
            "valid_records": records_per_expert,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        manifest_files[expert] = {
            "path": name,
            "records": records_per_expert,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "source_sha256": digest,
        }
        digest_parts.append(f"{expert}:{name}:{digest}:{records_per_expert}")
    task_bank_path = dataset_dir / "task_bank.jsonl"
    task_bank_path.write_text(
        "".join(
            json.dumps(
                {
                    "alignment_id": f"alignment-{index}",
                    "card_id": f"card-{index}",
                }
            )
            + "\n"
            for index in range(records_per_expert)
        ),
        encoding="utf-8",
    )
    task_bank_digest = hashlib.sha256(task_bank_path.read_bytes()).hexdigest()
    task_bank_source = {
        "path": "task_bank.jsonl",
        "records": records_per_expert,
        "bytes": task_bank_path.stat().st_size,
        "sha256": task_bank_digest,
    }
    task_bank_file = {**task_bank_source, "source_sha256": task_bank_digest}
    digest_parts.append(
        f"task_bank:task_bank.jsonl:{task_bank_digest}:{records_per_expert}"
    )
    snapshot_sha = hashlib.sha256("\n".join(digest_parts).encode()).hexdigest()
    calibration_count = max(1, records_per_expert // 10)
    calibration_dir = dataset_dir / "calibration"
    calibration_dir.mkdir()
    calibration_files: dict[str, dict] = {}
    for expert in REQUIRED_EXPERTS:
        calibration_path = calibration_dir / FILE_NAMES[expert]
        calibration_path.write_text(
            f"calibration-fixture:{expert}\n", encoding="utf-8"
        )
        calibration_files[expert] = {
            "path": f"calibration/{FILE_NAMES[expert]}",
            "records": calibration_count,
            "bytes": calibration_path.stat().st_size,
            "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": "anchor.training-snapshot.v2",
        "source_partition_manifest_sha256": "a" * 64,
        "source_gate": {
            "lineage_complete": True,
            "complete_chain_count": records_per_expert,
            "minimum_complete_chain_count": records_per_expert,
            "complete_chain_count_sufficient": True,
            "lineage_edge_error_count": 0,
            "lineage_chain_error_count": 0,
            "near_duplicate_gate": {"passed": True},
            "task_card_coverage": {
                "passed": True,
                "cardinality_equal": True,
                "complete_chain_count": records_per_expert,
                "card_count": records_per_expert,
                "unique_alignment_id_count": records_per_expert,
            },
            "task_bank_file": task_bank_source,
            "gold_files": {
                TASK_NAMES[expert]: {
                    "path": entry["path"],
                    "records": entry["records"],
                    "bytes": entry["bytes"],
                    "sha256": entry["source_sha256"],
                }
                for expert, entry in manifest_files.items()
            },
        },
        "snapshot_sha256": snapshot_sha,
        "population_contract": {
            "candidate_tasks_per_stage": CANDIDATE_TASKS_PER_STAGE,
            "candidate_work_orders": CANDIDATE_WORK_ORDERS,
            "work_orders_per_task": 5,
            "gold_accepted_tasks": records_per_expert + calibration_count,
        },
        "split_contract": {
            "schema_version": "anchor.formal-v3-gold-splits.v1",
            "assignment": "source_bank_split_then_gold_gate_v1",
            "pairwise_disjoint": True,
            "gold_coverage_complete": True,
            "heldout_content_read": False,
            "heldout_content_emitted": False,
            "leakage_audit_sha256": "b" * 64,
            "partitions": {
                "train": {
                    "role": "training_only",
                    "source_partition": "train",
                    "candidate_task_count": 17_105,
                    "gold_task_count": records_per_expert,
                    "ids_sha256": "c" * 64,
                    "gold_records_per_expert": {
                        expert: records_per_expert for expert in REQUIRED_EXPERTS
                    },
                },
                "calibration": {
                    "role": "rank_allocation_only",
                    "source_partition": "validation-from-train",
                    "candidate_task_count": 1_903,
                    "gold_task_count": calibration_count,
                    "ids_sha256": "d" * 64,
                    "snapshot_sha256": "c" * 64,
                    "gold_records_per_expert": {
                        expert: calibration_count for expert in REQUIRED_EXPERTS
                    },
                    "files": calibration_files,
                },
                "heldout": {
                    "role": "evaluation_only_hash_metadata",
                    "source_partition": "external-heldout",
                    "content_present": False,
                    "content_read": False,
                    "content_emitted": False,
                    "ids_sha256": "e" * 64,
                    "manifest_sha256": "f" * 64,
                },
            },
        },
        "task_bank_file": task_bank_file,
        "files": manifest_files,
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (dataset_dir / "manifest.json.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="ascii"
    )
    config = load_training_config(COMMON)
    return config, {"reports": reports}


def test_immutable_snapshot_manifest_verifies_every_binding(tmp_path: Path) -> None:
    config, datasets = _snapshot_fixture(tmp_path)
    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)
    assert report["passed"] is True
    assert report["computed_snapshot_sha256"] == report["declared_snapshot_sha256"]
    assert all(all(checks.values()) for checks in report["file_checks"].values())


def test_immutable_snapshot_manifest_rejects_legacy_128_record_snapshot(
    tmp_path: Path,
) -> None:
    config, datasets = _snapshot_fixture(tmp_path, records_per_expert=128)

    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)

    assert report["passed"] is False
    assert "snapshot source_gate lineage proof is invalid" in report["errors"]
    assert all(
        checks["minimum_records"] is False for checks in report["file_checks"].values()
    )


def test_immutable_snapshot_manifest_requires_lineage_proof(tmp_path: Path) -> None:
    config, datasets = _snapshot_fixture(tmp_path)
    manifest_path = tmp_path / "artifacts/formal_v3/dataset/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("source_gate")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (manifest_path.parent / "manifest.json.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="ascii"
    )

    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)

    assert report["passed"] is False
    assert "snapshot source_gate lineage proof is missing" in report["errors"]


def test_immutable_snapshot_manifest_rejects_source_gold_binding_tamper(
    tmp_path: Path,
) -> None:
    config, datasets = _snapshot_fixture(tmp_path)
    manifest_path = tmp_path / "artifacts/formal_v3/dataset/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_gate"]["gold_files"]["frontend"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (manifest_path.parent / "manifest.json.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="ascii"
    )

    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)

    assert report["passed"] is False
    assert report["file_checks"]["frontend_gen"]["source_gate_gold_binding"] is False
    assert "snapshot manifest binding failed for frontend_gen" in report["errors"]


def test_immutable_snapshot_manifest_rejects_sidecar_or_file_drift(
    tmp_path: Path,
) -> None:
    config, datasets = _snapshot_fixture(tmp_path)
    sidecar = tmp_path / "artifacts/formal_v3/dataset/manifest.json.sha256"
    sidecar.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")
    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)
    assert report["passed"] is False
    assert "snapshot manifest SHA-256 sidecar mismatch" in report["errors"]


def test_immutable_snapshot_rejects_heldout_content_access(tmp_path: Path) -> None:
    config, datasets = _snapshot_fixture(tmp_path)
    manifest_path = tmp_path / "artifacts/formal_v3/dataset/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_contract"]["heldout_content_read"] = True
    manifest["split_contract"]["partitions"]["heldout"]["content_read"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (manifest_path.parent / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)

    assert report["passed"] is False
    assert report["split_contract"]["heldout_content_read"] is False
    assert "snapshot split proof is invalid" in report["errors"]


def test_immutable_snapshot_rejects_more_gold_than_source_population(
    tmp_path: Path,
) -> None:
    config, datasets = _snapshot_fixture(tmp_path)
    manifest_path = tmp_path / "artifacts/formal_v3/dataset/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["population_contract"]["gold_accepted_tasks"] = 19_009
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (manifest_path.parent / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    report = inspect_dataset_snapshot_manifest(config, tmp_path, datasets)

    assert report["passed"] is False
    assert "snapshot candidate/Gold population contract is invalid" in report["errors"]


def test_launcher_is_safe_by_default_and_serializes_five_specialists() -> None:
    launcher = (ROOT / "scripts/train/run_formal_v3_lowmem.ps1").read_text(
        encoding="utf-8"
    )
    assert '[string]$Arm = "preflight"' in launcher
    assert "[switch]$Execute" in launcher
    assert "formal-v3-training.lock" in launcher
    assert "SKIP verified completed job" in launcher
    assert "Arm $ExpectedArm requires -AllocationManifest" in launcher
    assert "foreach ($Entry in $Ranks.GetEnumerator())" in launcher
    assert "partial/stale output exists" in launcher
    assert "automatic exact resume is not supported" in launcher
    assert 'ValidateSet("preflight", "smoke", "probe", "A", "B", "C"' in launcher
    assert "New-SnapshotSizedConfig" in launcher
    assert "Arm A is the frozen native Q4 baseline" in launcher


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None, reason="PowerShell required"
)
def test_adaptive_manifest_exact_experts_and_training_lock_cleanup(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    script_dir = project / "scripts" / "train"
    script_dir.mkdir(parents=True)
    launcher = script_dir / "run_formal_v3_lowmem.ps1"
    shutil.copy2(ROOT / "scripts/train/run_formal_v3_lowmem.ps1", launcher)
    snapshot_dir = project / "artifacts" / "formal_v3" / "dataset"
    snapshot_dir.mkdir(parents=True)
    snapshot_sha = "b" * 64
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": snapshot_sha,
                "split_contract": {
                    "partitions": {
                        "calibration": {"snapshot_sha256": "c" * 64}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    fake_python = tmp_path / "fake-python.cmd"
    fingerprint = "d" * 64
    fake_python.write_text(
        "@echo off\r\n"
        f'if "%1"=="-c" echo {{"fingerprint":"{fingerprint}",'
        '"max_steps":32,"rank":16,"alpha":32,'
        '"target_modules":["q_proj","v_proj"]}\r\n'
        "exit /b 0\r\n",
        encoding="ascii",
    )
    allocation = tmp_path / "allocation.json"
    base = {
        "schema_version": "anchor.lora-allocation.v1",
        "arm": "F",
        "dataset_snapshot_sha256": snapshot_sha,
        "mechanism_id": "stage_complexity_calibration_pareto_v1",
        "base_contract_id": "gemma4-12b-r56820d7-bnb-nf4-doublequant-bf16-v1",
        "target_modules": ["q_proj", "v_proj"],
            "parameters_per_rank": 649_216,
            "selection_status": "calibration_selected_frozen",
            "calibration_metrics_available": True,
            "calibration_record_count": 5,
            "selection_algorithm": {
                "algorithm_id": "measured-calibration-pareto-v1",
                "calibration_performance_used": True,
            },
        "calibration_snapshot_sha256": "c" * 64,
        "created_at": "2026-07-13T00:00:00+00:00",
        "allocation_frozen_at": "2026-07-13T00:01:00+00:00",
        "allocation_frozen_before_heldout": True,
        "heldout_access": "forbidden_until_allocation_frozen",
        "heldout_opened": False,
        "heldout_opened_at": None,
        "materialized_trainable_parameters": 10_387_456,
        "selection_objectives": [
            "maximize_per_stage_calibration_quality",
            "minimize_routed_latency",
            "minimize_peak_vram",
        ],
        "selected_ranks": {
            "planner": 3,
            "tool_policy": 3,
            "frontend_gen": 4,
            "frontend_review": 3,
            "security_gate": 3,
        },
    }
    base["attempted_allocations"] = [
        {
            "selected_ranks": dict(base["selected_ranks"]),
            "calibration_metrics": {"quality": 0.5},
        }
    ]

    def write_allocation(value: dict) -> None:
        allocation.write_text(json.dumps(value), encoding="utf-8")
        digest = hashlib.sha256(allocation.read_bytes()).hexdigest()
        Path(f"{allocation}.sha256").write_text(
            f"{digest}  {allocation.name}\n", encoding="ascii"
        )

    write_allocation(base)
    lock = tmp_path / "training.lock"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        "-Arm",
        "F",
        "-Execute",
        "-AllocationManifest",
        str(allocation),
        "-Python",
        str(fake_python),
        "-LockPath",
        str(lock),
    ]
    passed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert passed.returncode == 0, passed.stderr
    assert not lock.exists()

    base["selected_ranks"]["unexpected"] = 1
    write_allocation(base)
    rejected = subprocess.run(command, text=True, capture_output=True, check=False)
    assert rejected.returncode != 0
    assert "selected_ranks must name exactly the five specialists" in (
        rejected.stdout + rejected.stderr
    )
    assert not lock.exists()

    del base["selected_ranks"]["unexpected"]
    base["arm"] = "E"
    base["selection_objectives"] = [
        "maximize_per_stage_calibration_quality",
        "minimize_materialized_parameters",
        "minimize_routed_latency",
        "minimize_peak_vram",
    ]
    base["selected_ranks"] = {expert: 4 for expert in FILE_NAMES}
    base["attempted_allocations"] = [{"selected_ranks": dict(base["selected_ranks"])}]
    base["materialized_trainable_parameters"] = 649_216 * 20
    write_allocation(base)
    uniform_command = list(command)
    uniform_command[uniform_command.index("F")] = "E"
    uniform = subprocess.run(
        uniform_command, text=True, capture_output=True, check=False
    )
    assert uniform.returncode != 0
    assert "Arm E requires a non-uniform adaptive rank allocation" in (
        uniform.stdout + uniform.stderr
    )
    assert not lock.exists()

    base["selected_ranks"]["frontend_gen"] = 6
    base["attempted_allocations"] = [{"selected_ranks": dict(base["selected_ranks"])}]
    base["materialized_trainable_parameters"] = 649_216 * 22
    base["mechanism_id"] = "forged-mechanism"
    write_allocation(base)
    forged = subprocess.run(
        uniform_command, text=True, capture_output=True, check=False
    )
    assert forged.returncode != 0
    assert "not frozen to this formal-v3 snapshot/arm" in (
        forged.stdout + forged.stderr
    )
    assert not lock.exists()

    run_manifest = project / "artifacts/formal_v3/C/manifests/planner-r16.execute.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(
        json.dumps(
            {
                "mode": "execute",
                "run_name": "planner-r16",
                "config_sha256": fingerprint,
                "preflight": {
                    "passed": True,
                    "dataset_snapshot_sha256": snapshot_sha,
                    "dataset_snapshot_manifest": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    partial = project / "artifacts/formal_v3/C/adapters/planner-r16"
    partial.mkdir(parents=True)
    (partial / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    (partial / "adapter_model.safetensors").write_bytes(b"adapter")
    metadata_path = partial / "checkpoint_metadata.json"
    metadata = {
        "run_name": "planner-r16",
        "adapter_name": "planner",
        "config_sha256": fingerprint,
        "global_step": 32,
        "trainable_parameters": 10_387_456,
        "artifact_type": "peft_adapter",
        "merge_status": "unmerged",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    partial_command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        "-Arm",
        "C",
        "-Execute",
        "-Python",
        str(fake_python),
        "-LockPath",
        str(lock),
    ]
    completed = subprocess.run(
        partial_command, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "SKIP verified completed job: planner rank 16" in completed.stdout
    assert not lock.exists()

    metadata["global_step"] = 31
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    partial_run = subprocess.run(
        partial_command, text=True, capture_output=True, check=False
    )
    assert partial_run.returncode != 0
    assert "automatic exact resume is not supported" in (
        partial_run.stdout + partial_run.stderr
    )
    assert not lock.exists()

    shutil.rmtree(partial)
    run_manifest.unlink()
    progress = project / "artifacts/formal_v3/C/adapters/planner-r16.progress"
    progress.mkdir(parents=True)
    (progress / "status.json").write_text("{}\n", encoding="utf-8")
    progress_only = subprocess.run(
        partial_command, text=True, capture_output=True, check=False
    )
    assert progress_only.returncode != 0
    assert "planner-r16.progress" in (progress_only.stdout + progress_only.stderr)
    assert not lock.exists()
