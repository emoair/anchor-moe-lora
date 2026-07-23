from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training import cli  # noqa: E402


CONFIG = ROOT / "configs" / "training" / "gemma4_12b_qlora_smoke.yaml"
ONE_STEP_CONFIG = ROOT / "configs" / "training" / "gemma4_12b_qlora_one_step.yaml"
FORMAL_CONFIG = ROOT / "configs" / "training" / "formal_v3_lowmem_common.yaml"


def dependency_fixture(*, ready: bool) -> dict:
    return {
        "python": "3.10-test",
        "python_supported": True,
        "minimum_python": "3.10",
        "platform": "test",
        "packages": {},
        "missing": [] if ready else ["torch"],
        "incompatible": [],
        "device": {
            "probed": True,
            "cuda_available": ready,
            "bf16_supported": ready,
            "free_memory_gib": 11.5 if ready else None,
            "name": "Fake GPU" if ready else None,
        },
        "host_memory": {
            "probed": True,
            "available_memory_gib": 16.0 if ready else 4.0,
            "total_memory_gib": 24.0,
        },
        "ready": ready,
    }


def blocked_preflight() -> tuple[dict, list]:
    return (
        {
            "passed": False,
            "gates": {"three_live_datasets_present": {"passed": False, "evidence": {}}},
            "dataset_snapshot_sha256": "missing-data",
            "base": {},
            "heldout": {},
        },
        [],
    )


def passed_preflight() -> tuple[dict, list]:
    return (
        {
            "passed": True,
            "gates": {"all": {"passed": True, "evidence": {}}},
            "dataset_snapshot_sha256": "complete-live-data",
            "base": {"passed": True},
            "heldout": {"passed": True},
        },
        [
            {
                "id": "heldout-frontend",
                "expert": "frontend_gen",
                "prompt": "Probe frontend.",
                "max_new_tokens": 4,
            }
        ],
    )


def test_dry_run_never_imports_runtime_or_requires_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "dependency_report",
        lambda **_kwargs: dependency_fixture(ready=False),
    )
    sys.modules.pop("anchor_mvp.training.runtime", None)
    output = tmp_path / "manifest.json"
    result = cli.main(
        [
            "--config",
            str(CONFIG),
            "--adapter",
            "frontend_gen",
            "--rank",
            "16",
            "--dry-run",
            "--manifest-out",
            str(output),
        ]
    )
    assert result == 0
    assert "anchor_mvp.training.runtime" not in sys.modules
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["mode"] == "dry-run"
    assert manifest["base_model"] == "google/gemma-4-12B"
    assert manifest["base_model_revision"] == "56820d7d8cbe8e47975a53325439ed272e91cff2"
    assert manifest["training_precision"]["base_weights"] == (
        "training-compatible prequantized 4-bit checkpoint (frozen)"
    )
    assert manifest["training_precision"]["load_strategy"] == "prequantized_peft_4bit"
    assert manifest["training_profile"]["max_steps"] == 8
    # The resumable live corpus may exist in a developer checkout but is untracked on CI.
    # Dry-run must remain valid in both states and must not import the heavy runtime.
    assert isinstance(manifest["datasets"][0]["exists"], bool)
    if manifest["datasets"][0]["exists"]:
        assert manifest["datasets"][0]["ok"] is True


def test_preflight_writes_blocked_manifest_without_importing_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli, "dependency_report", lambda **_kwargs: dependency_fixture(ready=True)
    )
    monkeypatch.setattr(
        cli,
        "build_preflight_report",
        lambda config, root, dependencies, deep_checksum=False: blocked_preflight(),
    )
    sys.modules.pop("anchor_mvp.training.runtime", None)
    output = tmp_path / "preflight.json"
    result = cli.main(
        [
            "preflight",
            "--config",
            str(CONFIG),
            "--dry-run",
            "--manifest-out",
            str(output),
        ]
    )
    assert result == 3
    assert "anchor_mvp.training.runtime" not in sys.modules
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["stage"] == "preflight"
    assert manifest["preflight"]["passed"] is False


def test_execute_cannot_start_before_three_live_datasets_pass(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli, "dependency_report", lambda **_kwargs: dependency_fixture(ready=True)
    )
    monkeypatch.setattr(
        cli,
        "build_preflight_report",
        lambda config, root, dependencies, deep_checksum=False: blocked_preflight(),
    )
    sys.modules.pop("anchor_mvp.training.runtime", None)
    output = tmp_path / "blocked-execute.json"
    result = cli.main(
        [
            "train",
            "--config",
            str(CONFIG),
            "--adapter",
            "frontend_gen",
            "--execute",
            "--manifest-out",
            str(output),
        ]
    )
    assert result == 2
    assert "anchor_mvp.training.runtime" not in sys.modules
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["preflight"]["passed"] is False


def test_formal_execute_gate_precedes_gpu_data_manifest_and_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from anchor_mvp.training import formal_execution_gate as gate

    decision: dict[str, object] = {
        "schema_version": gate.FORMAL_DECISION_SCHEMA,
        "status": "blocked_formal_authorization_inputs_unavailable",
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "authenticated_inputs": {"formal_v3": "complete"},
    }
    decision["decision_sha256"] = gate._canonical_sha256(decision)
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", lambda: decision)

    def forbidden(*_args, **_kwargs):
        pytest.fail("formal execution touched a resource before its launcher lease")

    monkeypatch.setattr(cli, "dependency_report", forbidden)
    monkeypatch.setattr(cli, "build_preflight_report", forbidden)
    monkeypatch.setattr(cli, "_dataset_reports", forbidden)
    monkeypatch.setattr(cli, "build_manifest", forbidden)
    monkeypatch.setattr(cli, "write_json", forbidden)
    sys.modules.pop("anchor_mvp.training.runtime", None)
    output = tmp_path / "must-not-exist.json"

    result = cli.main(
        [
            "train",
            "--config",
            str(FORMAL_CONFIG),
            "--adapter",
            "planner",
            "--execute",
            "--manifest-out",
            str(output),
        ]
    )

    assert result == 2
    assert "v1 is blocked-only" in capsys.readouterr().err
    assert not output.exists()
    assert "anchor_mvp.training.runtime" not in sys.modules


@pytest.mark.parametrize("formal_component", ["FORMAL_V3", "formal_v3:stream"])
def test_formal_manifest_override_cannot_be_hidden_by_renaming_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    formal_component: str,
) -> None:
    from anchor_mvp.training import formal_execution_gate as gate

    decision: dict[str, object] = {
        "schema_version": gate.FORMAL_DECISION_SCHEMA,
        "status": "blocked_formal_authorization_inputs_unavailable",
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "authenticated_inputs": {"formal_v3": "complete"},
    }
    decision["decision_sha256"] = gate._canonical_sha256(decision)
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", lambda: decision)

    def forbidden(*_args, **_kwargs):
        pytest.fail("formal manifest override touched execution resources")

    monkeypatch.setattr(cli, "dependency_report", forbidden)
    monkeypatch.setattr(cli, "build_preflight_report", forbidden)
    monkeypatch.setattr(cli, "_dataset_reports", forbidden)
    monkeypatch.setattr(cli, "build_manifest", forbidden)
    monkeypatch.setattr(cli, "write_json", forbidden)
    sys.modules.pop("anchor_mvp.training.runtime", None)
    output = tmp_path / "ARTIFACTS" / formal_component
    if ":" not in formal_component:
        output /= "renamed.json"

    result = cli.main(
        [
            "train",
            "--config",
            str(CONFIG),
            "--adapter",
            "frontend_gen",
            "--execute",
            "--manifest-out",
            str(output),
        ]
    )

    assert result == 2
    assert not output.exists()
    assert "anchor_mvp.training.runtime" not in sys.modules


def test_one_step_smoke_gate_dry_run_records_profile_without_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli, "dependency_report", lambda **_kwargs: dependency_fixture(ready=True)
    )
    monkeypatch.setattr(
        cli,
        "build_preflight_report",
        lambda config, root, dependencies, deep_checksum=False: passed_preflight(),
    )
    sys.modules.pop("anchor_mvp.training.runtime", None)
    output = tmp_path / "smoke-dry-run.json"
    result = cli.main(
        [
            "smoke-gate",
            "--config",
            str(ONE_STEP_CONFIG),
            "--adapter",
            "frontend_gen",
            "--dry-run",
            "--manifest-out",
            str(output),
        ]
    )
    assert result == 0
    assert "anchor_mvp.training.runtime" not in sys.modules
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["stage"] == "smoke-gate"
    assert manifest["training_profile"]["max_steps"] == 1
    assert manifest["training_profile"]["max_seq_length"] == 64
    assert manifest["smoke_gate"] == {
        "executed": False,
        "ready": True,
        "passed": False,
    }
