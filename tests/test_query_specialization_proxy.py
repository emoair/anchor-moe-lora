from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from anchor_mvp.research.query_specialization import ROLES


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/research/train_query_specialization_mvp.py"
CONFIG_PATH = REPO_ROOT / "configs/research/query_specialization_mvp.yaml"
FIXTURE_ROOT = REPO_ROOT / "fixtures/research/taskboard_projector"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("query_specialization_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proxy_task_groups_are_deterministic_and_disjoint() -> None:
    module = _load_probe_module()
    proxy = {"train_tasks": 3, "eval_tasks": 2}

    first = module.proxy_task_ids(proxy)
    second = module.proxy_task_ids(dict(reversed(tuple(proxy.items()))))

    assert first == second
    assert set(first["train"]).isdisjoint(first["eval"])
    assert len(first["train"]) == 3
    assert len(first["eval"]) == 2


def test_every_synthetic_task_has_all_roles_and_both_variants() -> None:
    torch = pytest.importorskip("torch")
    module = _load_probe_module()
    task_ids = {"train": ("train-1",), "eval": ("eval-1",)}
    role_kinds = {role: ("requirement",) for role in ROLES}

    examples = module._build_proxy_examples(
        torch,
        task_ids=task_ids,
        role_kinds=role_kinds,
        width=16,
        task_variation=0.05,
        distractor_copies=2,
    )

    assert {example.task_id for example in examples["train"]} == {"train-1"}
    assert {example.task_id for example in examples["eval"]} == {"eval-1"}
    for split in ("train", "eval"):
        task_examples = examples[split]
        assert {example.role for example in task_examples} == set(ROLES)
        for role in ROLES:
            assert {
                example.variant for example in task_examples if example.role == role
            } == {"clean", "noisy"}


def test_dry_run_uses_non_promotional_claim_language() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    serialized = json.dumps(payload)

    assert payload["ok"] is True
    assert payload["plan"]["claim_scope"] == "experimental_proxy_scaffold_only"
    assert payload["plan"]["proxy"]["task_overlap"] == 0
    assert payload["plan"]["producer_manifest"]["counts"]["total"] == 15
    assert payload["plan"]["producer_manifest"]["provider_requests"] == 0
    assert payload["plan"]["producer_manifest"]["canonical_gold_written"] is False
    assert payload["plan"]["full_model_followup_enabled"] is False
    assert payload["plan"]["formal_training_authorized"] is False
    assert payload["plan"]["release_lock_status"] == "unavailable"
    assert payload["plan"]["release_lock_manifest_sha256"] is None
    assert (
        payload["plan"]["release_lock_validation"]["reason"]
        == "real_frozen_formal_v3_release_unavailable"
    )
    assert "promoted" not in serialized
    assert "promotion_checks" not in serialized


def test_execute_result_uses_versioned_content_hashes_and_signal_gate() -> None:
    pytest.importorskip("torch")
    module = _load_probe_module()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["proxy"].update(
        {
            "hidden_size": 16,
            "rank": 2,
            "alpha": 4,
            "train_tasks": 2,
            "eval_tasks": 1,
            "epochs": 1,
        }
    )
    _, records, _, _ = module._load_contract_dataset(config)

    result = module._execute_probe(
        config,
        records,
        experiment_config_sha256="a" * 64,
        contract_fixture_sha256="b" * 64,
        metrics_schema_sha256="c" * 64,
        runner_sha256="d" * 64,
        query_contract_module_sha256="e" * 64,
    )
    serialized = json.dumps(result)

    assert result["schema_version"] == "anchor.query-specialization-proxy-metrics.v1"
    assert result["experiment_config_sha256"] == "a" * 64
    assert result["contract_fixture_sha256"] == "b" * 64
    assert result["metrics_schema_sha256"] == "c" * 64
    assert result["runner_sha256"] == "d" * 64
    assert result["query_contract_module_sha256"] == "e" * 64
    assert result["contract_fixture_used_for_gradient_training"] is False
    assert result["metrics_scope"] == "unseen_eval_task_groups_only"
    assert result["proxy_signal_passed"] == all(result["signal_checks"].values())
    assert "promoted" not in serialized


def test_experiment_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["proxy"]["learnig_rate"] = 0.1
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(bad_config), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "proxy contains unknown fields" in completed.stderr


def test_enabling_full_model_without_ready_release_fails_closed(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["full_model_followup"]["enabled"] = True
    bad_config = tmp_path / "enabled-without-release.yaml"
    bad_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(bad_config), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "release_lock.status=ready" in completed.stderr


def test_contract_loader_rejects_tampered_sidecar_file(tmp_path: Path) -> None:
    copied = tmp_path / "taskboard_projector"
    shutil.copytree(FIXTURE_ROOT, copied)
    target = copied / "train/clean.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dataset_contract"]["root"] = str(copied)
    config["dataset_contract"]["manifest"] = str(copied / "manifest.json")
    bad_config = tmp_path / "tampered.yaml"
    bad_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(bad_config), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "mismatch" in completed.stderr


def test_contract_loader_rejects_wrong_pinned_schema_hash(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dataset_contract"]["expected_sidecar_schema_sha256"] = "0" * 64
    bad_config = tmp_path / "wrong-schema-hash.yaml"
    bad_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(bad_config), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "sidecar_schema hash mismatch" in completed.stderr
