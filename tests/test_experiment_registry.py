from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor_mvp.training.experiment_registry import (
    BUDGET_PARAMETERS,
    ExperimentRegistryError,
    index_completed_group,
    initialize_run,
    register_base_group,
    reserve_output,
    verify_registry,
)


RUN_ID = "formal-partial-v1-forced-20260715-v1"
SNAPSHOT_SHA = "2" * 64
PER_RANK_PARAMETERS = BUDGET_PARAMETERS // 16
ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    base = _write_json(
        root / "models/nf4/anchor_quantization_manifest.json",
        {
            "schema_version": "anchor.bnb-nf4-export.v1",
            "source_weight_sha256": "1" * 64,
            "weights": [
                {"path": "model-00001.safetensors", "bytes": 7, "sha256": "3" * 64}
            ],
        },
    )
    dataset = _write_json(
        root / "artifacts/dataset/manifest.json",
        {
            "schema_version": "anchor.per-expert-partial-training-snapshot.v1",
            "snapshot_sha256": SNAPSHOT_SHA,
            "not_for_end_to_end_claim": True,
        },
    )
    (root / "configs").mkdir()
    (root / "configs/benchmark.json").write_text("{}\n", encoding="utf-8")
    legacy = root / "artifacts/formal_partial_v1/forced_low_host_memory"
    legacy.mkdir(parents=True)
    return root, base, dataset


def _init(tmp_path: Path, *, layout: str = "legacy-layout-v1") -> tuple[Path, Path]:
    root, base, dataset = _project(tmp_path)
    registry_parent = (
        root / "artifacts/formal_partial_v1/forced_low_host_memory/registries"
    )
    legacy = root / "artifacts/formal_partial_v1/forced_low_host_memory"
    run_root = initialize_run(
        root,
        registry_parent,
        run_id=RUN_ID,
        profile="forced_low_host_memory",
        layout=layout,
        base_manifest=base,
        dataset_manifest=dataset,
        legacy_artifact_parent=legacy if layout == "legacy-layout-v1" else None,
    )
    return root, run_root


def _make_adapter(
    root: Path,
    *,
    group: str,
    adapter: str,
    rank: int,
    step: int = 32,
) -> None:
    legacy = root / "artifacts/formal_partial_v1/forced_low_host_memory" / group
    artifact_name = f"{adapter}-r{rank}"
    adapter_dir = legacy / "adapters" / artifact_name
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model = adapter_dir / "adapter_model.safetensors"
    model.write_bytes(f"{adapter}-{rank}".encode())
    _write_json(adapter_dir / "adapter_config.json", {"r": rank})
    config_path = root / f"configs/{group}-{adapter}.yaml"
    config_path.write_text(
        f"group: {group}\nadapter: {adapter}\nrank: {rank}\n", encoding="utf-8"
    )
    config_sha = hashlib.sha256(
        f"resolved-{group}-{adapter}-{rank}".encode()
    ).hexdigest()
    parameters = PER_RANK_PARAMETERS * rank
    _write_json(
        adapter_dir / "checkpoint_metadata.json",
        {
            "run_name": artifact_name,
            "adapter_name": adapter,
            "config_sha256": config_sha,
            "global_step": step,
            "trainable_parameters": parameters,
        },
    )
    _write_json(
        legacy / "manifests" / f"{artifact_name}.execute.json",
        {
            "mode": "execute",
            "stage": "train",
            "run_name": artifact_name,
            "adapter_name": adapter,
            "config_sha256": config_sha,
            "config_path": str(config_path),
            "preflight": {"dataset_snapshot_sha256": SNAPSHOT_SHA},
        },
    )
    _write_json(
        legacy / "adapters" / f"{artifact_name}.progress" / "status.json",
        {
            "state": "completed",
            "step": step,
            "run_id": hashlib.md5(artifact_name.encode()).hexdigest(),  # noqa: S324
        },
    )


def test_init_creates_isolated_a_through_f_and_registers_base_only_a(
    tmp_path: Path,
) -> None:
    root, run_root = _init(tmp_path)

    assert [path.name for path in run_root.iterdir() if path.is_dir()] == list("ABCDEF")
    registry = register_base_group(
        root, run_root, config_path=root / "configs/benchmark.json"
    )
    value = json.loads(registry.read_text(encoding="utf-8"))

    assert value["run_id"] == RUN_ID
    assert value["group"] == "A"
    assert value["training_performed"] is False
    assert value["adapters"] == []
    assert value["adapter_summary"]["trainable_parameter_total"] == 0
    assert value["base_artifact"]["manifest_sha256"]
    assert value["dataset_snapshot"]["snapshot_sha256"] == SNAPSHOT_SHA
    assert verify_registry(root, run_root, group="A")["ok"] is True
    assert not (run_root / "A/adapters").exists()
    assert not (run_root / "A/manifests").exists()


def test_run_and_group_registries_are_exclusive_and_never_overwritten(
    tmp_path: Path,
) -> None:
    root, run_root = _init(tmp_path)
    register_base_group(root, run_root, config_path=root / "configs/benchmark.json")
    before = (run_root / "A/group_registry.json").read_bytes()

    with pytest.raises(ExperimentRegistryError, match="refusing overwrite"):
        register_base_group(root, run_root, config_path=root / "configs/benchmark.json")
    with pytest.raises(ExperimentRegistryError, match="choose a new run_id"):
        initialize_run(
            root,
            run_root.parent,
            run_id=RUN_ID,
            profile="forced_low_host_memory",
            layout="legacy-layout-v1",
            base_manifest=root / "models/nf4/anchor_quantization_manifest.json",
            dataset_manifest=root / "artifacts/dataset/manifest.json",
            legacy_artifact_parent=root
            / "artifacts/formal_partial_v1/forced_low_host_memory",
        )

    assert (run_root / "A/group_registry.json").read_bytes() == before


def test_indexes_existing_c_without_mutating_or_moving_checkpoints(
    tmp_path: Path,
) -> None:
    root, run_root = _init(tmp_path)
    for adapter in (
        "planner",
        "tool_policy",
        "frontend_gen",
        "frontend_review",
        "security_gate",
    ):
        _make_adapter(root, group="C", adapter=adapter, rank=16)
    source = root / "artifacts/formal_partial_v1/forced_low_host_memory/C"
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    registry = index_completed_group(root, run_root, group="C")
    value = json.loads(registry.read_text(encoding="utf-8"))
    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert value["layout"] == "legacy-layout-v1"
    assert value["source"]["mode"] == "immutable-external"
    assert value["source"]["read_only_index"] is True
    assert value["adapter_summary"]["count"] == 5
    assert value["adapter_summary"]["rank_total"] == 80
    assert value["adapter_summary"]["steps_total"] == 160
    assert (
        value["adapter_summary"]["trainable_parameter_total"] == 5 * BUDGET_PARAMETERS
    )
    assert set(value["adapter_summary"]["adapter_sha256"]) == {
        "planner",
        "tool_policy",
        "frontend_gen",
        "frontend_review",
        "security_gate",
    }
    assert verify_registry(root, run_root, group="C")["ok"] is True


def test_verify_detects_post_registration_adapter_mutation(tmp_path: Path) -> None:
    root, run_root = _init(tmp_path)
    for adapter in (
        "planner",
        "tool_policy",
        "frontend_gen",
        "frontend_review",
        "security_gate",
    ):
        _make_adapter(root, group="C", adapter=adapter, rank=16)
    index_completed_group(root, run_root, group="C")
    model = (
        root
        / "artifacts/formal_partial_v1/forced_low_host_memory/C/adapters/planner-r16/adapter_model.safetensors"
    )
    model.write_bytes(b"mutated")

    with pytest.raises(ExperimentRegistryError, match="indexed adapter file changed"):
        verify_registry(root, run_root, group="C")


def test_reservation_rejects_a_duplicates_and_existing_output(tmp_path: Path) -> None:
    root, run_root = _init(tmp_path)

    with pytest.raises(ExperimentRegistryError, match="base-only"):
        reserve_output(root, run_root, group="A", artifact_name="planner-r16")
    reservation = reserve_output(
        root, run_root, group="B", artifact_name="mixed_all-r16"
    )
    assert json.loads(reservation.read_text())["run_id"] == RUN_ID
    with pytest.raises(ExperimentRegistryError, match="refusing overwrite"):
        reserve_output(root, run_root, group="B", artifact_name="mixed_all-r16")

    stale = (
        root
        / "artifacts/formal_partial_v1/forced_low_host_memory/D/adapters/planner-r3"
    )
    stale.mkdir(parents=True)
    with pytest.raises(ExperimentRegistryError, match="refusing to overwrite"):
        reserve_output(root, run_root, group="D", artifact_name="planner-r3")


def test_versioned_v2_places_future_outputs_below_run_id(tmp_path: Path) -> None:
    root, run_root = _init(tmp_path, layout="versioned-layout-v2")
    run = json.loads((run_root / "run_manifest.json").read_text())

    assert run["groups"]["B"]["artifact_root"].endswith(f"/{RUN_ID}/B")
    assert (run_root / "B/adapters").is_dir()
    assert (run_root / "F/manifests").is_dir()
    assert not (run_root / "A/adapters").exists()


def test_legacy_layout_can_bind_versioned_e_f_external_roots(tmp_path: Path) -> None:
    root, base, dataset = _project(tmp_path)
    legacy = root / "artifacts/formal_partial_v1/forced_low_host_memory"
    run_root = initialize_run(
        root,
        legacy / "registries",
        run_id=RUN_ID,
        profile="forced_low_host_memory",
        layout="legacy-layout-v1",
        base_manifest=base,
        dataset_manifest=dataset,
        legacy_artifact_parent=legacy,
        group_artifact_roots={
            "E": legacy / "E/heuristic-calibration-pending-v1",
            "F": legacy / "F/heuristic-calibration-pending-v1",
        },
    )
    run = json.loads((run_root / "run_manifest.json").read_text())

    assert run["groups"]["D"]["artifact_root"].endswith("forced_low_host_memory/D")
    assert run["groups"]["E"]["artifact_root"].endswith(
        "forced_low_host_memory/E/heuristic-calibration-pending-v1"
    )
    assert run["groups"]["F"]["artifact_root"].endswith(
        "forced_low_host_memory/F/heuristic-calibration-pending-v1"
    )


def test_finalize_script_is_append_only_and_defaults_to_b_through_f() -> None:
    text = (ROOT / "scripts/train/finalize_af_registries.ps1").read_text(
        encoding="utf-8"
    )

    assert '[string[]]$Groups = @("B", "C", "D", "E", "F")' in text
    assert '"verify", "--run-root"' in text
    assert '"index", "--run-root"' in text
    assert "group_registry.json" in text
    assert "--force" not in text
    assert "run_formal_partial_v1_lowmem.ps1" not in text
