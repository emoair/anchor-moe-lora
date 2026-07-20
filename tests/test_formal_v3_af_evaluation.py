from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.benchmark.formal_v3_preflight import (
    FormalV3PreflightError,
    preflight,
)
from anchor_mvp.benchmark.formal_v3_registry import (
    B_PARAMETERS,
    PARAMETERS_PER_RANK,
    FormalV3RegistryError,
    finalize_bundle,
    inspect_readiness,
)
from anchor_mvp.training.config import load_training_config, select_adapter
from anchor_mvp.training.manifest import config_fingerprint


ROOT = Path(__file__).resolve().parents[1]
STAGE_NAMES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_sidecar(path: Path) -> None:
    Path(f"{path}.sha256").write_text(
        f"{_sha(path)}  {path.name}\n", encoding="ascii"
    )


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "project"
    root.mkdir()
    base_dir = root / "models/base-nf4"
    base_dir.mkdir(parents=True)
    weight = base_dir / "model.safetensors"
    weight.write_bytes(b"frozen-nf4-weight")
    base_manifest = base_dir / "anchor_quantization_manifest.json"
    _write_json(
        base_manifest,
        {
            "schema_version": "anchor.bnb-nf4-export.v1",
            "source_weight_sha256": "1" * 64,
            "quantization": {
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "weights": [
                {
                    "path": weight.name,
                    "bytes": weight.stat().st_size,
                    "sha256": _sha(weight),
                }
            ],
        },
    )

    heldout_manifest = root / "artifacts/benchmark/heldout/manifest.json"
    leak_audit = root / "artifacts/benchmark/heldout/leak_audit.json"
    _write_json(heldout_manifest, {"metadata": "hash-only"})
    _write_json(leak_audit, {"passed": True, "metadata": "hash-only"})
    snapshot_path = root / "artifacts/formal_v3/dataset/manifest.json"
    snapshot_sha = "2" * 64
    calibration_sha = "3" * 64
    _write_json(
        snapshot_path,
        {
            "schema_version": "anchor.training-snapshot.v2",
            "snapshot_sha256": snapshot_sha,
            "split_contract": {
                "schema_version": "anchor.formal-v3-gold-splits.v1",
                "pairwise_disjoint": True,
                "heldout_content_read": False,
                "heldout_content_emitted": False,
                "leakage_audit_sha256": _sha(leak_audit),
                "partitions": {
                    "calibration": {"snapshot_sha256": calibration_sha},
                    "heldout": {
                        "role": "evaluation_only_hash_metadata",
                        "content_present": False,
                        "content_read": False,
                        "content_emitted": False,
                        "ids_sha256": "4" * 64,
                        "manifest_sha256": _sha(heldout_manifest),
                    },
                },
            },
        },
    )
    _write_sidecar(snapshot_path)

    allocations = {
        "E": {
            "planner": 8,
            "tool_policy": 4,
            "frontend_gen": 16,
            "frontend_review": 12,
            "security_gate": 4,
        },
        "F": {
            "planner": 4,
            "tool_policy": 1,
            "frontend_gen": 6,
            "frontend_review": 4,
            "security_gate": 1,
        },
    }
    for arm, ranks in allocations.items():
        allocation = root / f"artifacts/formal_v3/allocations/{arm}.json"
        _write_json(
            allocation,
            {
                "schema_version": "anchor.lora-allocation.v1",
                "arm": arm,
                "dataset_snapshot_sha256": snapshot_sha,
                "calibration_snapshot_sha256": calibration_sha,
                "mechanism_id": "stage_complexity_calibration_pareto_v1",
                "selection_status": "calibration_selected_frozen",
                "calibration_metrics_available": True,
                "calibration_record_count": 5,
                "selection_algorithm": {
                    "algorithm_id": "measured-calibration-pareto-v1",
                    "calibration_performance_used": True,
                },
                "target_modules": ["q_proj", "v_proj"],
                "parameters_per_rank": PARAMETERS_PER_RANK,
                "allocation_frozen_before_heldout": True,
                "created_at": "2026-07-18T00:00:00+00:00",
                "allocation_frozen_at": "2026-07-18T00:01:00+00:00",
                "heldout_access": "forbidden_until_allocation_frozen",
                "heldout_opened": False,
                "heldout_opened_at": None,
                "attempted_allocations": [
                    {
                        "selected_ranks": ranks,
                        "calibration_metrics": {"quality": 0.5},
                    }
                ],
                "selected_ranks": ranks,
                "rank_sum": sum(ranks.values()),
                "materialized_trainable_parameters": PARAMETERS_PER_RANK
                * sum(ranks.values()),
            },
        )
        _write_sidecar(allocation)

    rank_sets = {
        "B": {"mixed_all": 16},
        "C": {expert: 16 for expert in STAGE_NAMES},
        "D": {
            "planner": 3,
            "tool_policy": 3,
            "frontend_gen": 4,
            "frontend_review": 3,
            "security_gate": 3,
        },
        **allocations,
    }
    template = load_training_config(ROOT / "configs/training/formal_v3_lowmem_common.yaml")
    template = {key: value for key, value in template.items() if not key.startswith("_")}
    for arm, ranks in rank_sets.items():
        schedule_path = root / f"artifacts/formal_v3/schedules/{snapshot_sha}/{arm}.json"
        schedule = json.loads(json.dumps(template))
        schedule["experiment"] = f"anchor-moe-lora-formal-v3-lowmem-{arm}"
        schedule["model"]["local_path"] = "models/base-nf4"
        schedule["scale_gate"]["training_artifact"]["local_path"] = (
            "models/base-nf4"
        )
        schedule["paths"]["project_root"] = str(root)
        schedule["paths"]["manifest_dir"] = f"artifacts/formal_v3/{arm}/manifests"
        schedule["paths"]["adapter_dir"] = f"artifacts/formal_v3/{arm}/adapters"
        schedule["training"]["max_steps"] = 2
        schedule["training"]["resolved_exposure"] = {
            "schema_version": "anchor.formal-v3-exposure-plan.v1",
            "arm": arm,
            "max_steps_per_adapter_job": 2,
            "control_invariant": (
                "equal_total_and_per_stage_sample_exposure_B_through_F"
            ),
            "dataset_snapshot_sha256": snapshot_sha,
            "snapshot_manifest_sha256": _sha(snapshot_path),
            "heldout_content_read": False,
            "evaluation_contract": {
                "schema_version": "anchor.formal-v3-af-evaluation.v1",
                "heldout_ids_sha256": "4" * 64,
                "heldout_manifest_sha256": _sha(heldout_manifest),
                "leakage_audit_sha256": _sha(leak_audit),
                "normalization": "A_equals_100",
                "formal_v2_artifacts_allowed": False,
            },
        }
        _write_json(schedule_path, schedule)
        _write_sidecar(schedule_path)
        loaded = load_training_config(schedule_path)
        for adapter_name, rank in ranks.items():
            fingerprint = config_fingerprint(select_adapter(loaded, adapter_name, rank))
            artifact_name = f"{adapter_name}-r{rank}"
            adapter_dir = root / f"artifacts/formal_v3/{arm}/adapters/{artifact_name}"
            _write_json(
                adapter_dir / "adapter_config.json",
                {"r": rank, "lora_alpha": 2 * rank, "target_modules": ["q_proj", "v_proj"]},
            )
            (adapter_dir / "adapter_model.safetensors").write_bytes(
                f"{arm}:{artifact_name}".encode()
            )
            _write_json(
                adapter_dir / "checkpoint_metadata.json",
                {
                    "run_name": artifact_name,
                    "adapter_name": adapter_name,
                    "config_sha256": fingerprint,
                    "global_step": 2,
                    "trainable_parameters": PARAMETERS_PER_RANK * rank,
                    "artifact_type": "peft_adapter",
                    "merge_status": "unmerged",
                },
            )
            _write_json(
                root / f"artifacts/formal_v3/{arm}/manifests/{artifact_name}.execute.json",
                {
                    "mode": "execute",
                    "stage": "train",
                    "run_name": artifact_name,
                    "adapter_name": adapter_name,
                    "config_path": str(schedule_path),
                    "config_sha256": fingerprint,
                    "preflight": {
                        "passed": True,
                        "dataset_snapshot_sha256": snapshot_sha,
                        "dataset_snapshot_manifest": {"passed": True},
                    },
                },
            )
            _write_json(
                root
                / f"artifacts/formal_v3/{arm}/adapters/{artifact_name}.progress/status.json",
                {"state": "completed", "step": 2, "run_id": "5" * 32},
            )

    control = root / "configs/benchmark/formal_v3_af_control.json"
    _write_json(
        control,
        {
            "schema_version": "anchor.formal-v3-af-finalizer-config.v1",
            "formal_version": "formal-v3",
            "base_contract_id": "gemma4-test-bnb-nf4-v1",
            "base_manifest": base_manifest.relative_to(root).as_posix(),
            "snapshot_manifest": snapshot_path.relative_to(root).as_posix(),
            "snapshot_sidecar": Path(f"{snapshot_path}.sha256").relative_to(root).as_posix(),
            "schedule_root": "artifacts/formal_v3/schedules",
            "artifact_roots": {
                arm: f"artifacts/formal_v3/{arm}" for arm in ("B", "C", "D", "E", "F")
            },
            "allocation_manifests": {
                arm: f"artifacts/formal_v3/allocations/{arm}.json" for arm in ("E", "F")
            },
            "heldout_metadata": {
                "manifest_path": heldout_manifest.relative_to(root).as_posix(),
                "leak_audit_path": leak_audit.relative_to(root).as_posix(),
            },
            "protocol": {
                "stages": ["planner", "tool_policy", "frontend", "review", "security"],
                "review_protocol": "verdict_v2",
                "max_review_cycles": 2,
                "max_tokens_per_call": 512,
                "sampling": {"temperature": 0.0, "top_p": 1.0},
            },
        },
    )
    return control, "formal-v3-test-001"


def test_missing_training_is_explicitly_blocked(tmp_path: Path) -> None:
    root = tmp_path / "project"
    control = root / "control.json"
    _write_json(
        control,
        {
            "schema_version": "anchor.formal-v3-af-finalizer-config.v1",
            "formal_version": "formal-v3",
            "snapshot_manifest": "artifacts/formal_v3/dataset/manifest.json",
            "snapshot_sidecar": "artifacts/formal_v3/dataset/manifest.json.sha256",
        },
    )
    report = inspect_readiness(control, root, version_id="formal-v3-missing-001")
    assert report["status"] == "BLOCKED"
    assert report["code"] == "training_incomplete"
    assert report["heldout_case_content_read"] is False
    assert report["gpu_started"] is False
    assert report["api_called"] is False


def test_finalize_and_preflight_dynamic_af_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control, version = _fixture(tmp_path)
    root = control.parents[2]
    forbidden = root / "configs/benchmark/heldout_cases_v1.jsonl"
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == forbidden.resolve():
            raise AssertionError("offline preflight opened heldout case content")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    output = root / f"artifacts/formal_v3/evaluation/registries/{version}"
    finalized = finalize_bundle(
        control, root, version_id=version, output_dir=output
    )
    assert finalized["status"] == "READY"
    result = preflight(output / "benchmark.json", root)
    assert result["status"] == "READY"
    assert result["heldout_case_content_read"] is False
    assert result["comparison_plan"]["A_index"] == 100
    runtime = result["runtime_bindings"]
    assert all(runtime["A"][stage]["adapter_dir"] is None for stage in runtime["A"])
    assert len({runtime["B"][stage]["adapter_dir"] for stage in runtime["B"]}) == 1
    for arm in ("C", "D", "E", "F"):
        assert len({runtime[arm][stage]["adapter_dir"] for stage in runtime[arm]}) == 5
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    by_arm = {item["registry_group"]: item for item in benchmark["baselines"]}
    assert by_arm["D"]["stage_adapter_ranks"] == {
        "planner": 3,
        "tool_policy": 3,
        "frontend": 4,
        "review": 3,
        "security": 3,
    }
    assert by_arm["E"]["stage_adapter_ranks"]["frontend"] == 16
    assert sum(by_arm["F"]["stage_adapter_ranks"].values()) == 16
    assert by_arm["F"]["adapter_trainable_parameters"] == B_PARAMETERS
    registry = json.loads((output / "registry.json").read_text(encoding="utf-8"))
    assert registry["snapshot"]["sidecar"]["path"].endswith(
        "manifest.json.sha256"
    )
    assert registry["allocations"]["E"]["selection_status"] == (
        "calibration_selected_frozen"
    )
    assert registry["allocations"]["F"]["selection_algorithm_id"] == (
        "measured-calibration-pareto-v1"
    )


def test_f_allocation_must_exactly_match_b(tmp_path: Path) -> None:
    control, version = _fixture(tmp_path)
    root = control.parents[2]
    allocation = root / "artifacts/formal_v3/allocations/F.json"
    value = json.loads(allocation.read_text(encoding="utf-8"))
    value["selected_ranks"]["planner"] = 6
    value["rank_sum"] = 18
    value["materialized_trainable_parameters"] = 18 * PARAMETERS_PER_RANK
    _write_json(allocation, value)
    _write_sidecar(allocation)
    report = inspect_readiness(control, root, version_id=version)
    assert report["status"] == "BLOCKED"
    assert report["code"] == "allocation_mismatch"


def test_old_heuristic_allocation_is_not_formal_v3(tmp_path: Path) -> None:
    control, version = _fixture(tmp_path)
    root = control.parents[2]
    allocation = root / "artifacts/formal_v3/allocations/E.json"
    value = json.loads(allocation.read_text(encoding="utf-8"))
    value["selection_status"] = "heuristic_preregistered_calibration_pending"
    value["calibration_metrics_available"] = False
    _write_json(allocation, value)
    _write_sidecar(allocation)
    report = inspect_readiness(control, root, version_id=version)
    assert report["status"] == "BLOCKED"
    assert report["code"] == "allocation_mismatch"


def test_preflight_rejects_snapshot_sidecar_tampering(tmp_path: Path) -> None:
    control, version = _fixture(tmp_path)
    root = control.parents[2]
    output = root / f"artifacts/formal_v3/evaluation/registries/{version}"
    finalize_bundle(control, root, version_id=version, output_dir=output)
    sidecar = root / "artifacts/formal_v3/dataset/manifest.json.sha256"
    sidecar.write_text(sidecar.read_text(encoding="ascii") + "# changed\n", encoding="ascii")
    with pytest.raises(FormalV3PreflightError) as caught:
        preflight(output / "benchmark.json", root)
    assert caught.value.code == "artifact_changed"


def test_preflight_rejects_version_bundle_tampering(tmp_path: Path) -> None:
    control, version = _fixture(tmp_path)
    root = control.parents[2]
    output = root / f"artifacts/formal_v3/evaluation/registries/{version}"
    finalize_bundle(control, root, version_id=version, output_dir=output)
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    benchmark["formal_version"] = "formal-v2"
    _write_json(output / "benchmark.json", benchmark)
    with pytest.raises(FormalV3PreflightError) as caught:
        preflight(output / "benchmark.json", root)
    assert caught.value.code in {"formal_v2_input_forbidden", "bundle_mismatch"}


def test_finalizer_refuses_overwrite_and_formal_v2_input(tmp_path: Path) -> None:
    control, version = _fixture(tmp_path)
    root = control.parents[2]
    output = root / f"artifacts/formal_v3/evaluation/registries/{version}"
    finalize_bundle(control, root, version_id=version, output_dir=output)
    with pytest.raises(FormalV3RegistryError) as caught:
        finalize_bundle(control, root, version_id=version, output_dir=output)
    assert caught.value.code == "immutable_output_exists"
    value = json.loads(control.read_text(encoding="utf-8"))
    value["artifact_roots"]["B"] = "artifacts/formal-v2/B"
    _write_json(control, value)
    report = inspect_readiness(control, root, version_id="formal-v3-new-001")
    assert report["status"] == "BLOCKED"
    assert report["code"] == "formal_v2_input_forbidden"


def test_powershell_launcher_is_safe_and_version_isolated() -> None:
    launcher = (ROOT / "scripts/benchmark/run_formal_v3_af.ps1").read_text(
        encoding="utf-8"
    )
    assert "[switch]$Finalize" in launcher
    assert "[switch]$Execute" in launcher
    assert "[switch]$Resume" in launcher
    assert "[switch]$AuthorizeHeldoutAccess" in launcher
    assert 'runs/formal-v3/evaluation/$VersionId' in launcher
    assert "formal-v3 A-F evaluation is BLOCKED" in launcher
    assert "--serial-runtime-lora" in launcher
    assert "--authorize-heldout-access" in launcher
    assert "formal-v2" in launcher
    assert "runs/formal-v3/evaluation/$VersionId" in launcher
