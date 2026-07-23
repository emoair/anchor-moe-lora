from __future__ import annotations

import copy
import hashlib
import json
import os
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


CONFIG_DIR = ROOT / "configs" / "training"
COMMON = CONFIG_DIR / "formal_v3_lowmem_common.yaml"
PROFILES = {
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
        assert profile["training"]["optim"] == "paged_adamw_8bit"
        assert profile["training"]["allow_tf32"] is True
        assert profile["training"]["maximum_training_peak_vram_gib"] == 9.0
        assert profile["paths"]["adapter_dir"] == f"artifacts/formal_v3/{arm}/adapters"
        snapshot = profile["scale_gate"]["dataset_snapshot"]
        assert snapshot == {
            "schema_version": "anchor.training-snapshot.v2",
            "manifest": "artifacts/formal_v3/dataset/manifest.json",
            "sidecar": "artifacts/formal_v3/dataset/manifest.json.sha256",
            "immutable": True,
            "minimum_records_per_expert": FORMAL_MINIMUM,
        }
        for paths in (
            profile["scale_gate"]["required_datasets"].values(),
            *(entry["datasets"] for entry in profile["adapters"].values()),
        ):
            assert all(
                value.startswith("artifacts/formal_v3/dataset/") for value in paths
            )


def test_formal_v3_exposure_budget_matches_B_and_five_specialists() -> None:
    mixed = load_training_config(PROFILES["B"])
    specialist = load_training_config(PROFILES["C"])
    mixed_exposures = (
        mixed["training"]["max_steps"]
        * mixed["training"]["gradient_accumulation_steps"]
    )
    routed_exposures = 5 * (
        specialist["training"]["max_steps"]
        * specialist["training"]["gradient_accumulation_steps"]
    )
    assert mixed_exposures == routed_exposures == 640


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


@pytest.mark.parametrize("rank", [1, 6, 8, 12, 16])
def test_adaptive_rank_menu_is_trainable(rank: int) -> None:
    selected = select_adapter(load_training_config(PROFILES["E"]), "planner", rank)
    assert selected["lora"]["rank"] == rank
    assert selected["lora"]["alpha"] == 2 * rank


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
    assert "anchor_mvp.research.formal_authorization_consumer" in launcher
    assert "anchor.formal-authorization-decision.v1" in launcher
    assert "decision v1 is permanently blocked" in launcher
    assert "versioned v2-or-later decision plus an authenticated execution lease" in (
        launcher
    )
    assert "LockPath must equal the canonical single-GPU lock" in launcher
    assert launcher.index("[IO.File]::Open(", launcher.index("try {")) < launcher.index(
        "$AuthorizationDecision = Invoke-FormalAuthorization"
    )
    assert launcher.index(
        "$AuthorizationDecision = Invoke-FormalAuthorization"
    ) < launcher.rindex("Invoke-Preflight $Config")


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None, reason="PowerShell required"
)
def test_adaptive_manifest_exact_experts_in_non_execute_validation(
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
        json.dumps({"snapshot_sha256": snapshot_sha}), encoding="utf-8"
    )
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "@echo off\r\nexit /b 0\r\n",
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
    base["attempted_allocations"] = [{"selected_ranks": dict(base["selected_ranks"])}]

    def write_allocation(value: dict) -> None:
        allocation.write_text(json.dumps(value), encoding="utf-8")
        digest = hashlib.sha256(allocation.read_bytes()).hexdigest()
        Path(f"{allocation}.sha256").write_text(
            f"{digest}  {allocation.name}\n", encoding="ascii"
        )

    write_allocation(base)
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        "-Arm",
        "F",
        "-AllocationManifest",
        str(allocation),
        "-Python",
        str(fake_python),
    ]
    passed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert passed.returncode == 0, passed.stderr

    base["selected_ranks"]["unexpected"] = 1
    write_allocation(base)
    rejected = subprocess.run(command, text=True, capture_output=True, check=False)
    assert rejected.returncode != 0
    assert "selected_ranks must name exactly the five specialists" in (
        rejected.stdout + rejected.stderr
    )

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


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None, reason="PowerShell required"
)
@pytest.mark.parametrize(
    "mode",
    [
        "blocked",
        "empty",
        "malformed",
        "wrong_schema",
        "false",
        "bad_sha",
        "ready_v1",
    ],
)
def test_execute_authorization_is_strict_and_cleans_lock(
    tmp_path: Path, mode: str
) -> None:
    project = tmp_path / "project"
    script_dir = project / "scripts" / "train"
    script_dir.mkdir(parents=True)
    launcher = script_dir / "run_formal_v3_lowmem.ps1"
    shutil.copy2(ROOT / "scripts/train/run_formal_v3_lowmem.ps1", launcher)
    fake_python = tmp_path / "fake-python.cmd"
    ready_sha = "e" * 64
    fake_python.write_text(
        "@echo off\r\n"
        'if not "%FAKE_PYTHON_LOG%"=="" echo %*>>"%FAKE_PYTHON_LOG%"\r\n'
        'if "%2"=="anchor_mvp.research.formal_authorization_consumer" goto auth\r\n'
        "exit /b 0\r\n"
        ":auth\r\n"
        'if not exist "%FAKE_EXPECT_LOCK%" exit /b 91\r\n'
        'if "%FAKE_FORMAL_AUTH_MODE%"=="empty" exit /b 0\r\n'
        'if "%FAKE_FORMAL_AUTH_MODE%"=="malformed" echo not-json& exit /b 0\r\n'
        'if "%FAKE_FORMAL_AUTH_MODE%"=="blocked" '
        'echo {"schema_version":"anchor.formal-authorization-decision.v1",'
        '"status":"blocked","formal_training_authorized":false,'
        f'"decision_sha256":"{ready_sha}"}}& exit /b 2\r\n'
        'if "%FAKE_FORMAL_AUTH_MODE%"=="wrong_schema" '
        'echo {"schema_version":"anchor.wrong.v1","status":"ready",'
        '"formal_training_authorized":true,'
        f'"decision_sha256":"{ready_sha}"}}& exit /b 0\r\n'
        'if "%FAKE_FORMAL_AUTH_MODE%"=="false" '
        'echo {"schema_version":"anchor.formal-authorization-decision.v1",'
        '"status":"ready","formal_training_authorized":false,'
        f'"decision_sha256":"{ready_sha}"}}& exit /b 0\r\n'
        'if "%FAKE_FORMAL_AUTH_MODE%"=="bad_sha" '
        'echo {"schema_version":"anchor.formal-authorization-decision.v1",'
        '"status":"ready","formal_training_authorized":true,'
        '"decision_sha256":"not-a-sha"}& exit /b 0\r\n'
        'echo {"schema_version":"anchor.formal-authorization-decision.v1",'
        '"status":"ready","formal_training_authorized":true,'
        f'"decision_sha256":"{ready_sha}"}}\r\n'
        "exit /b 0\r\n",
        encoding="ascii",
    )
    lock = project / "runs" / "formal-v3-training.lock"
    log = tmp_path / "python.log"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        "-Arm",
        "preflight",
        "-Execute",
        "-Python",
        str(fake_python),
        "-LockPath",
        str(lock),
    ]
    env = {
        **os.environ,
        "FAKE_FORMAL_AUTH_MODE": mode,
        "FAKE_EXPECT_LOCK": str(lock),
        "FAKE_PYTHON_LOG": str(log),
    }
    rejected = subprocess.run(
        command, text=True, capture_output=True, check=False, env=env
    )
    assert rejected.returncode != 0
    assert "formal-v3 authorization refused" in (rejected.stdout + rejected.stderr)
    assert not lock.exists()
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "anchor_mvp.research.formal_authorization_consumer" in invocations[0]
    assert "anchor_mvp.training" not in invocations[0]


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None, reason="PowerShell required"
)
def test_execute_rejects_noncanonical_lock_before_authorization(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    script_dir = project / "scripts" / "train"
    script_dir.mkdir(parents=True)
    launcher = script_dir / "run_formal_v3_lowmem.ps1"
    shutil.copy2(ROOT / "scripts/train/run_formal_v3_lowmem.ps1", launcher)
    fake_python = tmp_path / "fake-python.cmd"
    marker = tmp_path / "python-invoked.txt"
    fake_python.write_text(
        "@echo off\r\n"
        'if not "%FAKE_PYTHON_MARKER%"=="" echo invoked>"%FAKE_PYTHON_MARKER%"\r\n'
        "exit /b 0\r\n",
        encoding="ascii",
    )
    noncanonical_lock = tmp_path / "different-training.lock"
    canonical_lock = project / "runs" / "formal-v3-training.lock"
    rejected = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-Arm",
            "preflight",
            "-Execute",
            "-Python",
            str(fake_python),
            "-LockPath",
            str(noncanonical_lock),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "FAKE_PYTHON_MARKER": str(marker)},
    )
    assert rejected.returncode != 0
    assert "LockPath must equal the canonical single-GPU lock" in (
        rejected.stdout + rejected.stderr
    )
    assert not marker.exists()
    assert not noncanonical_lock.exists()
    assert not canonical_lock.exists()


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None, reason="PowerShell required"
)
def test_execute_forged_v1_ready_is_rejected_before_preflight_and_cleans_lock(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    script_dir = project / "scripts" / "train"
    script_dir.mkdir(parents=True)
    launcher = script_dir / "run_formal_v3_lowmem.ps1"
    shutil.copy2(ROOT / "scripts/train/run_formal_v3_lowmem.ps1", launcher)
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "@echo off\r\n"
        'if not "%FAKE_PYTHON_LOG%"=="" echo %*>>"%FAKE_PYTHON_LOG%"\r\n'
        'if "%2"=="anchor_mvp.research.formal_authorization_consumer" goto auth\r\n'
        "exit /b 0\r\n"
        ":auth\r\n"
        'if not exist "%FAKE_EXPECT_LOCK%" exit /b 91\r\n'
        'echo {"schema_version":"anchor.formal-authorization-decision.v1",'
        '"status":"ready","formal_training_authorized":true,'
        f'"decision_sha256":"{"e" * 64}"}}\r\n'
        "exit /b 0\r\n",
        encoding="ascii",
    )
    lock = project / "runs" / "formal-v3-training.lock"
    log = tmp_path / "python.log"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        "-Arm",
        "preflight",
        "-Execute",
        "-Python",
        str(fake_python),
        "-LockPath",
        str(lock),
    ]
    env = {
        **os.environ,
        "FAKE_FORMAL_AUTH_MODE": "ready",
        "FAKE_EXPECT_LOCK": str(lock),
        "FAKE_PYTHON_LOG": str(log),
    }
    rejected = subprocess.run(
        command, text=True, capture_output=True, check=False, env=env
    )
    assert rejected.returncode != 0
    assert "decision v1 is permanently blocked" in (rejected.stdout + rejected.stderr)
    assert not lock.exists()
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "anchor_mvp.research.formal_authorization_consumer" in invocations[0]
    assert "anchor_mvp.training preflight" not in invocations[0]

    log.unlink()
    dry_run_command = [item for item in command if item != "-Execute"]
    dry_run_env = {**env, "FAKE_FORMAL_AUTH_MODE": "blocked"}
    dry_run = subprocess.run(
        dry_run_command,
        text=True,
        capture_output=True,
        check=False,
        env=dry_run_env,
    )
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    assert not lock.exists()
    dry_run_invocations = log.read_text(encoding="utf-8").splitlines()
    assert len(dry_run_invocations) == 1
    assert "anchor_mvp.training preflight" in dry_run_invocations[0]
    assert "formal_authorization_consumer" not in dry_run_invocations[0]
