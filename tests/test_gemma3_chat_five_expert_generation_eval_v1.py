from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anchor_mvp.research import (
    gemma3_chat_five_expert_generation_eval_v1 as evaluator,
)


def _digest(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_with_sidecar(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    digest = _digest(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _bound_config() -> dict[str, object]:
    config = evaluator.load_config()
    config.pop("_config_sha256", None)
    config["identity_state"] = "authenticated_inputs"
    consumer = config["consumer"]
    assert isinstance(consumer, dict)
    consumer["config_sha256"] = "a" * 64
    consumer["implementation_sha256"] = "b" * 64
    consumer["dataset_manifest_sha256"] = "c" * 64
    partitions = consumer["partition_sha256"]
    assert isinstance(partitions, dict)
    partitions["train/chat.jsonl"] = "d" * 64
    partitions["eval_proxy/chat.jsonl"] = "e" * 64
    training = config["training_run"]
    assert isinstance(training, dict)
    training["run_id"] = "future-run"
    training["receipt_sha256"] = "f" * 64
    return config


def test_checked_in_config_binds_consumer_and_keeps_training_run_pending() -> None:
    config = evaluator.load_config()
    assert evaluator._pending(config) == (
        "training_run.run_id",
        "training_run.receipt_sha256",
    )
    status = evaluator.build_status(config)
    assert status["status"] == "blocked_waiting_for_authenticated_training_run"
    assert status["audit"]["dataset_body_reads"] == 0
    assert status["audit"]["model_loads"] == 0
    assert status["audit"]["gpu_requests"] == 0


def test_config_sidecar_is_mandatory_and_exact(tmp_path: Path) -> None:
    raw = evaluator.CONFIG_PATH.read_bytes()
    copied = tmp_path / evaluator.CONFIG_PATH.name
    copied.write_bytes(raw)
    with pytest.raises(
        evaluator.ConfigError,
        match="must be a physical file",
    ):
        evaluator.load_config(copied)
    copied.with_name(copied.name + ".sha256").write_bytes(
        f"{_digest(raw)}  wrong-name.json\n".encode("ascii")
    )
    with pytest.raises(
        evaluator.ChatGenerationEvalError,
        match="config_sidecar_invalid",
    ):
        evaluator.load_config(copied)


def test_pending_preflight_returns_before_source_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evaluator.load_config()

    def unexpected_authentication(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pending preflight must not authenticate sources")

    monkeypatch.setattr(evaluator, "authenticate_inputs", unexpected_authentication)
    result = evaluator.build_preflight(config)
    assert result["status"].startswith("blocked_waiting")
    assert result["audit"]["dataset_body_reads"] == 0
    assert result["audit"]["provider_requests"] == 0


def test_all_identities_are_conjunctive_and_wrong_route_is_a_derangement() -> None:
    bound = _bound_config()
    evaluator.validate_config(bound)
    wrong = bound["evaluation"]["wrong_route_adapter"]
    assert set(wrong) == set(evaluator.ROLES)
    assert set(wrong.values()) == set(evaluator.ROLES)
    assert all(wrong[role] != role for role in evaluator.ROLES)

    drifted = copy.deepcopy(bound)
    drifted["consumer"]["partition_sha256"]["eval_proxy/chat.jsonl"] = "pending"
    with pytest.raises(
        evaluator.ChatGenerationEvalError,
        match="pending_identity_state_drift",
    ):
        evaluator.validate_config(drifted)


def test_style_role_metrics_are_explicit_reference_proxies() -> None:
    result = evaluator.score_output(
        role="humor",
        generated_text="A tiny joke lands with a cheerful punchline.",
        reference_text="A tiny joke lands with a cheerful punchline!",
        generated_token_ids=(4, 5, 6, 7, 8, 9),
        eos_terminated=True,
        generation_seconds=0.25,
    )
    assert result["role_format_valid"] is True
    assert result["role_appropriate_proxy"] is True
    assert result["tool_closed_schema"] is False
    assert result["review_closed_schema"] is False
    assert result["repetition_gate_passed"] is True


def test_tool_schema_and_evidence_are_reference_bound() -> None:
    reference = {
        "tool": "local_search",
        "arguments": {"query": "status"},
        "evidence": {"ref_id": "result-1", "result": "ready"},
    }
    valid = evaluator.score_output(
        role="tool_call",
        generated_text=json.dumps(reference, sort_keys=True),
        reference_text=json.dumps(reference, sort_keys=True),
        generated_token_ids=(1, 2, 3, 4, 5, 6),
        eos_terminated=True,
        generation_seconds=0.5,
    )
    assert valid["tool_json_parse"] is True
    assert valid["tool_closed_schema"] is True
    assert valid["tool_evidence_consistency"] is True

    ungrounded = copy.deepcopy(reference)
    ungrounded["evidence"]["result"] = "invented"
    invalid = evaluator.score_output(
        role="tool_call",
        generated_text=json.dumps(ungrounded, sort_keys=True),
        reference_text=json.dumps(reference, sort_keys=True),
        generated_token_ids=(1, 2, 3, 4, 5, 6),
        eos_terminated=True,
        generation_seconds=0.5,
    )
    assert invalid["tool_closed_schema"] is True
    assert invalid["tool_evidence_consistency"] is False


def test_review_schema_and_dependency_consistency_are_reference_bound() -> None:
    reference = {
        "verdict": "pass",
        "dependency": {
            "record_id": "tool-record",
            "role": "tool_call",
            "target_sha256": "a" * 64,
        },
        "findings": [],
    }
    valid = evaluator.score_output(
        role="review_audit",
        generated_text=json.dumps(reference, sort_keys=True),
        reference_text=json.dumps(reference, sort_keys=True),
        generated_token_ids=(9, 8, 7, 6, 5, 4),
        eos_terminated=True,
        generation_seconds=0.4,
    )
    assert valid["review_json_parse"] is True
    assert valid["review_closed_schema"] is True
    assert valid["review_dependency_consistency"] is True

    mismatched = copy.deepcopy(reference)
    mismatched["dependency"]["record_id"] = "wrong-record"
    invalid = evaluator.score_output(
        role="review_audit",
        generated_text=json.dumps(mismatched, sort_keys=True),
        reference_text=json.dumps(reference, sort_keys=True),
        generated_token_ids=(9, 8, 7, 6, 5, 4),
        eos_terminated=True,
        generation_seconds=0.4,
    )
    assert invalid["review_closed_schema"] is True
    assert invalid["review_dependency_consistency"] is False


def test_persona_metric_requires_air_and_rejects_provider_misattribution() -> None:
    valid = evaluator.score_output(
        role="serious",
        generated_text="我是由Air训练的测试模型。",
        reference_text="我是由Air训练的测试模型。",
        generated_token_ids=(1, 2, 3, 4, 5),
        eos_terminated=True,
        generation_seconds=0.2,
    )
    assert valid["identity_probe"] is True
    assert valid["persona_identity_consistency"] is True
    assert valid["provider_misattribution_absent"] is True

    invalid = evaluator.score_output(
        role="serious",
        generated_text="I was trained by OpenAI, not Air.",
        reference_text="I am a test model trained by Air.",
        generated_token_ids=(1, 2, 3, 4, 5),
        eos_terminated=True,
        generation_seconds=0.2,
    )
    assert invalid["identity_probe"] is True
    assert invalid["persona_identity_consistency"] is False
    assert invalid["provider_misattribution_absent"] is False


def test_aggregate_receipt_has_no_bodies_tokens_or_per_record_metrics() -> None:
    metrics = evaluator.AggregateMetrics()
    metrics.add(
        evaluator.score_output(
            role="angry_style",
            generated_text="Firm response.",
            reference_text="Firm response!",
            generated_token_ids=(1, 2, 3, 4),
            eos_terminated=True,
            generation_seconds=0.1,
        )
    )
    receipt = {
        "metrics": {"angry_style": {"correct_adapter": metrics.report()}},
        "audit": {
            "generated_bodies_persisted": 0,
            "raw_token_ids_persisted": 0,
            "per_record_metrics_persisted": 0,
        },
    }
    evaluator._assert_aggregate_only_receipt(receipt)
    with pytest.raises(
        evaluator.ChatGenerationEvalError,
        match="body_or_per_record_field_forbidden",
    ):
        evaluator._assert_aggregate_only_receipt(
            {"metrics": {"per_record": [{"generated_text": "forbidden"}]}}
        )


def test_gpu_lock_writes_canonical_bytes_in_binary_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "formal-v3-training.lock"
    monkeypatch.setattr(evaluator, "LOCK_PATH", lock_path)
    monkeypatch.setattr(evaluator, "CONFLICTING_LOCKS", ())

    with evaluator._gpu_lock("eval-native-newline-test") as lock:
        raw = lock_path.read_bytes()
        assert raw.endswith(b"\n")
        assert not raw.endswith(b"\r\n")
        assert _digest(raw) == lock["content_sha256"]

    assert not lock_path.exists()


def test_main_reports_only_fixed_chat_error_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject(_path: str) -> dict[str, object]:
        raise evaluator.ChatGenerationEvalError("eval_gpu_lock_changed")

    monkeypatch.setattr(evaluator, "load_config", reject)
    assert evaluator.main(["--status", "--config", "unused.json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "ChatGenerationEvalError"
    assert payload["error_code"] == "eval_gpu_lock_changed"


def test_main_redacts_unregistered_chat_error_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject(_path: str) -> dict[str, object]:
        raise evaluator.ChatGenerationEvalError("eval_dynamic_secret_value")

    monkeypatch.setattr(evaluator, "load_config", reject)
    assert evaluator.main(["--status", "--config", "unused.json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_code"] == "runtime_error_redacted"


def test_main_redacts_arbitrary_non_chat_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "dynamic-sensitive-marker"

    def reject(_path: str) -> dict[str, object]:
        raise ValueError(marker)

    monkeypatch.setattr(evaluator, "load_config", reject)
    assert evaluator.main(["--status", "--config", "unused.json"]) == 1
    output = capsys.readouterr().out
    assert marker not in output
    payload = json.loads(output)
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_code"] == "runtime_error_redacted"


def test_wrong_route_deltas_use_role_primary_metrics() -> None:
    reports: dict[str, dict[str, dict[str, float | int]]] = {}
    for role in evaluator.ROLES:
        metric = (
            "tool_call_schema_valid_rate_all_records"
            if role == "tool_call"
            else "review_audit_schema_valid_rate_all_records"
            if role == "review_audit"
            else "role_appropriate_reference_proxy_rate"
        )
        reports[role] = {
            "base": {metric: 0.2},
            "correct_adapter": {metric: 0.8},
            "wrong_route": {metric: 0.3},
        }
    deltas = evaluator.build_route_deltas(reports)
    assert deltas["macro_correct_minus_wrong_route"] == 0.5
    assert deltas["macro_correct_minus_base"] == 0.6
    assert "not causal mechanism" in deltas["interpretation"]


def test_training_receipt_authenticates_all_adapters_and_phase_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    consumer = config["consumer"]
    training = config["training_run"]
    assert isinstance(consumer, dict)
    assert isinstance(training, dict)
    run_id = str(training["run_id"])
    run_root = (
        tmp_path
        / "artifacts"
        / "diagnostics"
        / "gemma3_1b_it_chat_five_expert_qonly_rank1024_v1"
        / run_id
    )
    role_entries = []
    for role in evaluator.ROLES:
        adapter = run_root / role / "adapter"
        adapter.mkdir(parents=True)
        adapter_hashes = {}
        for filename in evaluator.ADAPTER_FILES:
            raw = f"{role}:{filename}".encode("utf-8")
            (adapter / filename).write_bytes(raw)
            adapter_hashes[filename] = _digest(raw)
        smoke_sha = _write_with_sidecar(
            run_root / role / "smoke_receipt.json",
            {"role": role, "phase": "smoke"},
        )
        full_sha = _write_with_sidecar(
            run_root / role / "full_receipt.json",
            {"role": role, "phase": "full"},
        )
        role_entries.append(
            {
                "role": role,
                "smoke_receipt_sha256": smoke_sha,
                "full_receipt_sha256": full_sha,
                "adapter_artifact_sha256": adapter_hashes,
            }
        )
    receipt = {
        "schema_version": training["receipt_schema_version"],
        "status": training["required_status"],
        "run_id": run_id,
        "identity": {
            "config_sha256": consumer["config_sha256"],
            "dataset_manifest_sha256": consumer["dataset_manifest_sha256"],
            "partition_sha256": consumer["partition_sha256"],
        },
        "roles": role_entries,
        "claims": {
            "diagnostic_only": True,
            "training_executed": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
        },
    }
    training["artifact_root"] = (
        "artifacts/diagnostics/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1"
    )
    receipt_sha = _write_with_sidecar(run_root / "run_receipt.json", receipt)
    training["receipt_sha256"] = receipt_sha
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    dataset = SimpleNamespace(manifest_sha256=consumer["dataset_manifest_sha256"])
    consumer_config = {"_config_sha256": consumer["config_sha256"]}
    observed, observed_root, adapters, observed_sha = (
        evaluator._authenticate_training_run(
            config,
            consumer_config=consumer_config,
            dataset=dataset,
        )
    )
    assert observed["run_id"] == run_id
    assert observed_root == run_root
    assert set(adapters) == set(evaluator.ROLES)
    assert observed_sha == receipt_sha

    (run_root / "humor" / "adapter" / "adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(
        evaluator.ChatGenerationEvalError,
        match="adapter_file_sha_mismatch",
    ):
        evaluator._authenticate_training_run(
            config,
            consumer_config=consumer_config,
            dataset=dataset,
        )


def test_binding_helper_publishes_atomic_model_free_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evaluator.load_config()
    config_path = (
        tmp_path
        / "configs"
        / "training"
        / "gemma3_1b_it_chat_five_expert_qonly_rank1024_v1.yaml"
    )
    implementation_path = (
        tmp_path
        / "src"
        / "anchor_mvp"
        / "training"
        / "gemma3_chat_five_expert_qonly_v1.py"
    )
    config_path.parent.mkdir(parents=True)
    implementation_path.parent.mkdir(parents=True)
    config_path.write_bytes(b"consumer-config\n")
    implementation_path.write_bytes(b"consumer-implementation\n")
    config_sha = _digest(config_path.read_bytes())
    manifest_sha = "c" * 64
    partitions = {
        "train/chat.jsonl": "d" * 64,
        "eval_proxy/chat.jsonl": "e" * 64,
    }
    training_run_id = "finished-run"
    run_receipt_path = (
        tmp_path
        / "artifacts"
        / "diagnostics"
        / "gemma3_1b_it_chat_five_expert_qonly_rank1024_v1"
        / training_run_id
        / "run_receipt.json"
    )
    _write_with_sidecar(
        run_receipt_path,
        {
            "schema_version": evaluator.chat.RUN_RECEIPT_VERSION,
            "status": "passed_diagnostic_training_only",
            "run_id": training_run_id,
            "identity": {
                "config_sha256": config_sha,
                "dataset_manifest_sha256": manifest_sha,
                "partition_sha256": partitions,
            },
        },
    )
    dataset = SimpleNamespace(
        manifest_sha256=manifest_sha,
        partition_sha256=partitions,
    )
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluator.chat,
        "load_config",
        lambda _path: {"_config_sha256": config_sha},
    )
    monkeypatch.setattr(
        evaluator.chat,
        "_load_authenticated_dataset",
        lambda _config: dataset,
    )
    monkeypatch.setattr(
        evaluator,
        "_authenticate_training_run",
        lambda *_args, **_kwargs: ({}, run_receipt_path.parent, {}, "f" * 64),
    )
    published = evaluator.bind_training_run(
        config,
        training_run_id=training_run_id,
    )
    assert published.is_file()
    assert published.with_name(published.name + ".sha256").is_file()
    bound = evaluator.load_config(published)
    assert bound["identity_state"] == "authenticated_inputs"
    assert bound["consumer"]["dataset_manifest_sha256"] == manifest_sha
    assert bound["training_run"]["run_id"] == training_run_id


def test_one_click_launcher_exposes_status_preflight_execute() -> None:
    launcher = (
        evaluator.ROOT
        / "scripts"
        / "research"
        / "evaluate_gemma3_chat_five_expert_qonly_v1.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "[switch]$Status" in launcher
    assert "[switch]$Preflight" in launcher
    assert "[switch]$Execute" in launcher
    assert "[switch]$Bind" in launcher
    assert "[string]$TrainingRunId" in launcher
    assert '"--status"' in launcher
    assert '"--preflight"' in launcher
    assert '"--execute"' in launcher
    assert '"--bind-training-run"' in launcher
