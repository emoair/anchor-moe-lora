from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import (
    gemma3_chat_unbalanced_v2_generation_eval_v1 as evaluator,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bound_config() -> dict[str, object]:
    config = copy.deepcopy(evaluator.load_config())
    config.pop("_config_sha256")
    consumer = config["consumer_preflight"]
    training = config["training_run"]
    kv = config["kv_probe"]
    dataset = config["dataset_inventory"]
    assert isinstance(consumer, dict)
    assert isinstance(training, dict)
    assert isinstance(kv, dict)
    assert isinstance(dataset, dict)
    consumer["receipt_sha256"] = _digest("consumer-receipt")
    consumer["artifact_tree_digest_sha256"] = _digest("artifact-tree")
    consumer["shard_inventory_sha256"] = _digest("shard-inventory")
    training["receipt_sha256"] = _digest("training-receipt")
    training["adapter_inventory_sha256"] = _digest("adapter-inventory")
    training["run_id"] = "diagnostic-run"
    kv["receipt_sha256"] = _digest("kv-receipt")
    dataset["shard_inventory_sha256"] = consumer["shard_inventory_sha256"]
    evaluator.validate_config(config)
    return config


def _dependency_receipts(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    consumer = config["consumer_preflight"]
    training = config["training_run"]
    assert isinstance(consumer, dict)
    assert isinstance(training, dict)
    consumer_receipt = {
        "schema_version": evaluator.CONSUMER_RECEIPT_VERSION,
        "status": "passed",
        "identity": {
            "artifact_tree_digest_sha256": consumer["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": consumer["shard_inventory_sha256"],
        },
    }
    training_receipt = {
        "schema_version": evaluator.TRAINING_RECEIPT_VERSION,
        "status": "passed_diagnostic_multiarm_training",
        "run_id": training["run_id"],
        "identity": {
            "consumer_preflight_receipt_sha256": consumer["receipt_sha256"],
            "artifact_tree_digest_sha256": consumer["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": consumer["shard_inventory_sha256"],
            "adapter_inventory_sha256": training["adapter_inventory_sha256"],
        },
        "claims": {
            "fresh_base": True,
            "fresh_adapter": True,
            "resume": False,
        },
    }
    kv_receipt = {
        "schema_version": evaluator.KV_RECEIPT_VERSION,
        "status": "passed_diagnostic_prefix_value_handoff",
        "identity": {
            "consumer_preflight_receipt_sha256": consumer["receipt_sha256"],
            "shard_inventory_sha256": consumer["shard_inventory_sha256"],
        },
        "claims": {
            "exact_prefix_value_handoff": True,
            "prefill_compute_handoff": True,
            "shared_storage": False,
            "zero_copy": False,
            "full_generation_kv_shared": False,
            "rdma": False,
        },
    }
    return consumer_receipt, training_receipt, kv_receipt


ReceiptMutator = Callable[[dict[str, object]], None]


def _write_json(path: Path, receipt: dict[str, object]) -> str:
    raw = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _closed_schema(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(receipt),
        "properties": {name: {"const": value} for name, value in receipt.items()},
    }


def _physical_dependencies(
    tmp_path: Path,
    config: dict[str, object],
    *,
    mutate_consumer: ReceiptMutator | None = None,
    mutate_consumer_schema: ReceiptMutator | None = None,
    mutate_training: ReceiptMutator | None = None,
    mutate_training_schema: ReceiptMutator | None = None,
    mutate_kv: ReceiptMutator | None = None,
    mutate_kv_schema: ReceiptMutator | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    consumer_config = config["consumer_preflight"]
    training_config = config["training_run"]
    kv_config = config["kv_probe"]
    dataset_config = config["dataset_inventory"]
    assert isinstance(consumer_config, dict)
    assert isinstance(training_config, dict)
    assert isinstance(kv_config, dict)
    assert isinstance(dataset_config, dict)

    consumer, _, _ = _dependency_receipts(config)
    if mutate_consumer is not None:
        mutate_consumer(consumer)
    consumer_path = tmp_path / "consumer-receipt.json"
    consumer_config["receipt_sha256"] = _write_json(consumer_path, consumer)
    consumer_schema = _closed_schema(consumer)
    if mutate_consumer_schema is not None:
        mutate_consumer_schema(consumer_schema)
    consumer_schema_path = tmp_path / "consumer-receipt.schema.json"
    consumer_config["receipt_schema_sha256"] = _write_json(
        consumer_schema_path, consumer_schema
    )
    dataset_config["shard_inventory_sha256"] = consumer_config["shard_inventory_sha256"]

    _, training, kv = _dependency_receipts(config)
    if mutate_training is not None:
        mutate_training(training)
    if mutate_kv is not None:
        mutate_kv(kv)
    training_path = tmp_path / "training-receipt.json"
    kv_path = tmp_path / "kv-receipt.json"
    training_config["receipt_sha256"] = _write_json(training_path, training)
    training_schema = _closed_schema(training)
    if mutate_training_schema is not None:
        mutate_training_schema(training_schema)
    training_schema_path = tmp_path / "training-receipt.schema.json"
    training_config["receipt_schema_sha256"] = _write_json(
        training_schema_path, training_schema
    )
    kv_config["receipt_sha256"] = _write_json(kv_path, kv)
    kv_schema = _closed_schema(kv)
    if mutate_kv_schema is not None:
        mutate_kv_schema(kv_schema)
    kv_schema_path = tmp_path / "kv-receipt.schema.json"
    kv_config["receipt_schema_sha256"] = _write_json(kv_schema_path, kv_schema)
    evaluator.validate_config(config)
    return (
        consumer_path,
        consumer_schema_path,
        training_path,
        training_schema_path,
        kv_path,
        kv_schema_path,
    )


def _dependency_args(
    paths: tuple[Path, Path, Path, Path, Path, Path],
) -> dict[str, Path]:
    return {
        "consumer_receipt_path": paths[0],
        "consumer_receipt_schema_path": paths[1],
        "training_receipt_path": paths[2],
        "training_receipt_schema_path": paths[3],
        "kv_receipt_path": paths[4],
        "kv_receipt_schema_path": paths[5],
    }


def _materialized_case(
    tmp_path: Path,
) -> tuple[dict[str, object], tuple[Path, Path, Path, Path, Path, Path]]:
    config = _bound_config()
    return config, _physical_dependencies(tmp_path, config)


def _all_pass_ratio(records: int) -> dict[str, int | float]:
    return evaluator.ratio(records, records)


def _arm(records: int, metrics: tuple[str, ...]) -> dict[str, object]:
    return {
        "records": records,
        **{metric: _all_pass_ratio(records) for metric in metrics},
    }


def _performance(count: int) -> dict[str, int | float]:
    return {
        "count": count,
        "latency_ms_mean": 2.0,
        "latency_ms_p50": 1.5,
        "latency_ms_p95": 3.5,
        "throughput_units_per_second": 100.0,
    }


def _passed_receipt(config: dict[str, object]) -> dict[str, object]:
    consumer = config["consumer_preflight"]
    training = config["training_run"]
    kv = config["kv_probe"]
    assert isinstance(consumer, dict)
    assert isinstance(training, dict)
    assert isinstance(kv, dict)
    counts = evaluator.expected_count_matrix()
    receipt: dict[str, object] = {
        "schema_version": evaluator.RECEIPT_VERSION,
        "profile_id": evaluator.PROFILE_ID,
        "status": "passed_diagnostic_proxy_evaluation",
        "identity": {
            "config_sha256": evaluator.config_sha256(config),
            "implementation_sha256": evaluator.implementation_sha256(),
            "consumer_preflight_receipt_sha256": consumer["receipt_sha256"],
            "artifact_tree_digest_sha256": consumer["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": consumer["shard_inventory_sha256"],
            "training_run_receipt_sha256": training["receipt_sha256"],
            "adapter_inventory_sha256": training["adapter_inventory_sha256"],
            "kv_probe_receipt_sha256": kv["receipt_sha256"],
            "run_id": training["run_id"],
        },
        "planned_counts": counts,
        "observed_counts": copy.deepcopy(counts),
        "metrics": {
            "tool": {
                arm: _arm(400, evaluator.TOOL_METRICS) for arm in evaluator.TOOL_ARMS
            },
            "planner": {
                arm: _arm(240, evaluator.PLANNER_METRICS)
                for arm in evaluator.PLANNER_ARMS
            },
            "router": {
                "train_proxy": {
                    "records": 80,
                    "top1_accuracy": _all_pass_ratio(80),
                    "macro_f1": "1.000000000000",
                    "confusion_aggregate": [27, 0, 0, 0, 27, 0, 0, 0, 26],
                },
                "eval_proxy": {
                    "records": 20,
                    "top1_accuracy": _all_pass_ratio(20),
                    "macro_f1": "1.000000000000",
                    "confusion_aggregate": [7, 0, 0, 0, 7, 0, 0, 0, 6],
                },
            },
            "identity": {
                arm: _arm(50, evaluator.IDENTITY_METRICS)
                for arm in evaluator.IDENTITY_ARMS
            },
            "wrong_route": {
                arm: _arm(860, evaluator.WRONG_ROUTE_METRICS)
                for arm in evaluator.WRONG_ROUTE_DERANGEMENTS
            },
        },
        "strata": {
            "tool": {
                "seen_template_interpolation": {
                    "records": 160,
                    "primary_metric": _all_pass_ratio(160),
                },
                "argument_composition_proxy": {
                    "records": 120,
                    "primary_metric": _all_pass_ratio(120),
                },
                "cross_language_transfer_proxy": {
                    "records": 120,
                    "primary_metric": _all_pass_ratio(120),
                },
            },
            "planner": {
                name: {
                    "records": 80,
                    "primary_metric": _all_pass_ratio(80),
                }
                for name in (
                    "task_classification",
                    "expert_selection",
                    "commit_boundary",
                )
            },
        },
        "comparisons": {
            "tool_q_only_minus_base": "0.000000000000",
            "tool_q_plus_micro_o_minus_q_only": "0.000000000000",
            "planner_q_only_minus_base": "0.000000000000",
            "planner_o_only_minus_base": "0.000000000000",
            "planner_q_plus_micro_o_minus_q_only": "0.000000000000",
            "identity_correct_minus_base": "0.000000000000",
            "identity_correct_minus_wrong": "0.000000000000",
            "wrong_route_macro_degradation": "0.000000000000",
        },
        "performance": {
            "tool": _performance(1200),
            "planner": _performance(960),
            "router": _performance(100),
            "identity": _performance(150),
            "wrong_route": _performance(2580),
            "shared_prefix_kv": _performance(1),
        },
        "kv_handoff": {
            "receipt_status": "passed_diagnostic_prefix_value_handoff",
            "exact_prefix_value_handoff": True,
            "prefill_compute_handoff": True,
            "shared_storage": False,
            "zero_copy": False,
            "full_generation_kv_shared": False,
            "rdma": False,
            "tail_private_append_only": True,
            "layer_value_comparisons": 26,
        },
        "failure_codes": [],
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "quality_validated": False,
            "generalization_validated": False,
            "eval_proxy_is_heldout": False,
            "wrong_route_is_causal_proof": False,
            "kv_exact_prefix_value_observed": True,
            "kv_shared_storage_claimed": False,
            "kv_zero_copy_claimed": False,
            "kv_rdma_claimed": False,
            "physical_execution_completed": True,
            "formal": False,
            "release": False,
        },
        "audit": {
            "sample_bodies_persisted": 0,
            "raw_token_ids_persisted": 0,
            "per_record_arrays_persisted": 0,
            "record_identifiers_persisted": 0,
            "gold_reads": 0,
            "heldout_body_reads": 0,
            "protected_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
        },
    }
    metrics = receipt["metrics"]
    assert isinstance(metrics, dict)
    receipt["comparisons"] = evaluator.expected_comparisons(metrics)
    return receipt


def test_checked_in_contract_is_pending_sharded_and_model_free() -> None:
    config = evaluator.load_config()
    status = evaluator.build_status(config)
    assert status["status"] == "blocked_waiting_for_authenticated_inputs"
    assert status["expected_counts"]["total_slots"] == 4990
    assert config["consumer_preflight"]["required_schema_version"].endswith(
        ".sharded-v1"
    )
    assert config["dataset_inventory"]["physical_paths_hardcoded"] is False
    serialized = json.dumps(config, sort_keys=True)
    assert "train/chat.jsonl" not in serialized
    assert "physical_file_count" not in serialized
    assert "candidate_manifest_sha256" not in serialized
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source


def test_both_schemas_are_valid_draft_2020_12() -> None:
    for path in (evaluator.CONFIG_SCHEMA_PATH, evaluator.RECEIPT_SCHEMA_PATH):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_complete_matrix_counts_and_strata_are_exact() -> None:
    config = evaluator.load_config()
    matrix = config["matrix"]
    assert matrix["tool"]["records"] * len(matrix["tool"]["arms"]) == 1200
    assert sum(matrix["tool"]["strata"].values()) == 400
    assert matrix["planner"]["records"] * len(matrix["planner"]["arms"]) == 960
    assert sum(matrix["planner"]["strata"].values()) == 240
    assert matrix["router"] == {"train": 80, "eval_proxy": 20, "slots": 100}
    assert matrix["identity"]["records"] * len(matrix["identity"]["arms"]) == 150
    assert (
        matrix["wrong_route"]["records"] * len(matrix["wrong_route"]["derangements"])
        == 2580
    )
    assert (
        sum(
            (
                matrix["tool"]["slots"],
                matrix["planner"]["slots"],
                matrix["router"]["slots"],
                matrix["identity"]["slots"],
                matrix["wrong_route"]["slots"],
            )
        )
        == 4990
    )


def test_old_consumer_receipt_version_is_rejected(tmp_path: Path) -> None:
    config = _bound_config()

    def old_version(receipt: dict[str, object]) -> None:
        receipt["schema_version"] = (
            "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.v1"
        )

    paths = _physical_dependencies(
        tmp_path,
        config,
        mutate_consumer=old_version,
    )
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(
            config,
            **_dependency_args(paths),
        )


def test_authenticated_dependency_metadata_makes_preflight_ready(
    tmp_path: Path,
) -> None:
    config = _bound_config()
    paths = _physical_dependencies(tmp_path, config)
    preflight = evaluator.build_preflight(
        config,
        **_dependency_args(paths),
    )
    assert preflight["ready"] is True
    assert preflight["status"] == "ready_for_lazy_physical_backend"
    assert preflight["audit"]["dataset_body_reads"] == 0


def test_parsed_mapping_cannot_authenticate_a_dependency(tmp_path: Path) -> None:
    config = _bound_config()
    paths = _physical_dependencies(tmp_path, config)
    consumer, _, _ = _dependency_receipts(config)
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(
            config,
            consumer_receipt_path=consumer,
            consumer_receipt_schema_path=paths[1],
            training_receipt_path=paths[2],
            training_receipt_schema_path=paths[3],
            kv_receipt_path=paths[4],
            kv_receipt_schema_path=paths[5],
        )


@pytest.mark.parametrize(
    ("target", "error_code"),
    [
        ("consumer", "eval_consumer_preflight_pending"),
        ("training", "eval_training_run_pending"),
        ("kv", "eval_kv_probe_pending"),
    ],
)
@pytest.mark.parametrize(
    "mutation",
    (
        "extra",
        "missing",
        "wrong_version",
        "wrong_type",
        "nested_extra",
        "nested_missing",
        "nested_type",
    ),
)
def test_dependency_closed_contract_rejects_shape_drift(
    tmp_path: Path,
    target: str,
    mutation: str,
    error_code: str,
) -> None:
    config = _bound_config()

    def mutate(value: dict[str, object]) -> None:
        if mutation == "extra":
            value["unexpected"] = True
        elif mutation == "missing":
            value.pop("status")
        elif mutation == "wrong_version":
            value["schema_version"] = "wrong.version"
        elif mutation == "wrong_type":
            value["status"] = 1
        else:
            identity = value["identity"]
            assert isinstance(identity, dict)
            first = next(iter(identity))
            if mutation == "nested_extra":
                identity["unexpected"] = True
            elif mutation == "nested_missing":
                identity.pop(first)
            else:
                identity[first] = 1

    kwargs = {f"mutate_{target}": mutate}
    paths = _physical_dependencies(tmp_path, config, **kwargs)
    with pytest.raises(evaluator.GenerationEvalError, match=error_code):
        evaluator.build_preflight(
            config,
            **_dependency_args(paths),
        )


def test_physical_sha_and_terminal_identity_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    paths = _physical_dependencies(tmp_path, config)
    paths[0].write_bytes(b"{}\n")
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(
            config,
            **_dependency_args(paths),
        )

    config = _bound_config()
    paths = _physical_dependencies(
        tmp_path / "terminal-drift",
        config,
    )
    original_signature = evaluator._stat_signature
    calls = 0

    def drift_on_terminal(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = original_signature(value)
        if calls == 4:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(evaluator, "_stat_signature", drift_on_terminal)
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(
            config,
            **_dependency_args(paths),
        )


@pytest.mark.parametrize(
    ("missing_index", "error_code"),
    [
        (0, "eval_consumer_preflight_pending"),
        (1, "eval_consumer_preflight_pending"),
        (2, "eval_training_run_pending"),
        (3, "eval_training_run_pending"),
        (4, "eval_kv_probe_pending"),
        (5, "eval_kv_probe_pending"),
    ],
)
def test_every_dependency_receipt_and_schema_is_mandatory(
    tmp_path: Path, missing_index: int, error_code: str
) -> None:
    config, paths = _materialized_case(tmp_path)
    arguments = _dependency_args(paths)
    arguments[tuple(arguments)[missing_index]] = None
    with pytest.raises(evaluator.GenerationEvalError, match=error_code):
        evaluator.build_preflight(config, **arguments)


@pytest.mark.parametrize("target", ("consumer", "training", "kv"))
@pytest.mark.parametrize("mutation", ("wrong_version", "open_shape", "extra_key"))
def test_dependency_schema_contract_is_physically_authenticated(
    tmp_path: Path, target: str, mutation: str
) -> None:
    config = _bound_config()

    def mutate(schema: dict[str, object]) -> None:
        if mutation == "wrong_version":
            properties = schema["properties"]
            assert isinstance(properties, dict)
            schema_version = properties["schema_version"]
            assert isinstance(schema_version, dict)
            schema_version["const"] = "wrong.version"
        elif mutation == "open_shape":
            schema["additionalProperties"] = True
        else:
            schema["unexpected"] = True

    paths = _physical_dependencies(
        tmp_path,
        config,
        **{f"mutate_{target}_schema": mutate},
    )
    error_code = {
        "consumer": "eval_consumer_preflight_pending",
        "training": "eval_training_run_pending",
        "kv": "eval_kv_probe_pending",
    }[target]
    with pytest.raises(evaluator.GenerationEvalError, match=error_code):
        evaluator.build_preflight(config, **_dependency_args(paths))


def test_dependency_schema_sha_and_terminal_identity_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _materialized_case(tmp_path / "sha")
    paths[1].write_bytes(b"{}\n")
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(config, **_dependency_args(paths))

    config, paths = _materialized_case(tmp_path / "terminal")
    original_signature = evaluator._stat_signature
    calls = 0

    def drift_on_terminal(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = original_signature(value)
        if calls == 8:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(evaluator, "_stat_signature", drift_on_terminal)
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(config, **_dependency_args(paths))


@pytest.mark.parametrize("path_index", (0, 1))
@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_version":NaN}\n',
        b'{"schema_version":"first","schema_version":"second"}\n',
    ),
)
def test_dependency_receipt_and_schema_reject_nonfinite_and_duplicate_json(
    tmp_path: Path, path_index: int, raw: bytes
) -> None:
    config, paths = _materialized_case(tmp_path)
    paths[path_index].write_bytes(raw)
    dependency = config["consumer_preflight"]
    assert isinstance(dependency, dict)
    field = "receipt_sha256" if path_index == 0 else "receipt_schema_sha256"
    dependency[field] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.build_preflight(config, **_dependency_args(paths))


def test_public_receipt_validation_cannot_bypass_dependencies(tmp_path: Path) -> None:
    config, _ = _materialized_case(tmp_path)
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.validate_receipt(_passed_receipt(config), config)


def test_training_and_kv_claim_mutations_fail_closed(tmp_path: Path) -> None:
    config = _bound_config()

    def resumed(receipt: dict[str, object]) -> None:
        claims = receipt["claims"]
        assert isinstance(claims, dict)
        claims["resume"] = True

    paths = _physical_dependencies(
        tmp_path / "training",
        config,
        mutate_training=resumed,
    )
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_training_run_pending"
    ):
        evaluator.build_preflight(
            config,
            **_dependency_args(paths),
        )

    config = _bound_config()

    def upgrades_storage(receipt: dict[str, object]) -> None:
        claims = receipt["claims"]
        assert isinstance(claims, dict)
        claims["shared_storage"] = True

    paths = _physical_dependencies(
        tmp_path / "kv",
        config,
        mutate_kv=upgrades_storage,
    )
    with pytest.raises(evaluator.GenerationEvalError, match="eval_kv_probe_pending"):
        evaluator.build_preflight(
            config,
            **_dependency_args(paths),
        )


def test_passed_receipt_is_aggregate_only_and_exactly_counted(
    tmp_path: Path,
) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    validated = evaluator.validate_receipt(receipt, config, **_dependency_args(paths))
    assert validated["observed_counts"]["tool_slots"] == 1200
    assert validated["observed_counts"]["planner_slots"] == 960
    assert validated["observed_counts"]["wrong_route_slots"] == 2580
    assert validated["claims"]["quality_validated"] is False
    assert validated["claims"]["generalization_validated"] is False
    assert validated["claims"]["eval_proxy_is_heldout"] is False


def test_canonical_sorted_receipt_validates_through_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, paths = _materialized_case(tmp_path)
    config_path = tmp_path / "config.json"
    receipt_path = tmp_path / "receipt.json"
    _write_json(config_path, config)
    _write_json(receipt_path, _passed_receipt(config))

    result = evaluator.main(
        [
            "--config",
            str(config_path),
            "--consumer-receipt",
            str(paths[0]),
            "--consumer-receipt-schema",
            str(paths[1]),
            "--training-receipt",
            str(paths[2]),
            "--training-receipt-schema",
            str(paths[3]),
            "--kv-receipt",
            str(paths[4]),
            "--kv-receipt-schema",
            str(paths[5]),
            "--validate-receipt",
            str(receipt_path),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == (
        "passed_diagnostic_proxy_evaluation"
    )


def test_planner_arms_and_strata_mapping_order_is_semantically_irrelevant(
    tmp_path: Path,
) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    planner = receipt["metrics"]["planner"]
    strata = receipt["strata"]
    assert isinstance(planner, dict)
    assert isinstance(strata, dict)

    receipt["metrics"]["planner"] = dict(reversed(tuple(planner.items())))
    for group in ("tool", "planner"):
        summaries = strata[group]
        assert isinstance(summaries, dict)
        strata[group] = dict(reversed(tuple(summaries.items())))
    receipt["strata"] = dict(reversed(tuple(strata.items())))

    validated = evaluator.validate_receipt(
        receipt,
        config,
        **_dependency_args(paths),
    )
    assert validated["status"] == "passed_diagnostic_proxy_evaluation"


@pytest.mark.parametrize("target", ("planner_arms", "planner_strata"))
@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_planner_arms_and_strata_require_exact_key_sets(
    tmp_path: Path,
    target: str,
    mutation: str,
) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    mapping = (
        receipt["metrics"]["planner"]
        if target == "planner_arms"
        else receipt["strata"]["planner"]
    )
    assert isinstance(mapping, dict)
    if mutation == "extra":
        mapping["unexpected"] = copy.deepcopy(next(iter(mapping.values())))
    else:
        mapping.pop(next(iter(mapping)))

    with pytest.raises(evaluator.GenerationEvalError, match="eval_receipt_invalid"):
        evaluator.validate_receipt(
            receipt,
            config,
            **_dependency_args(paths),
        )


def test_rate_and_count_drift_fail_closed(tmp_path: Path) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    receipt["metrics"]["tool"]["tool_q_only"]["closed_schema"]["rate"] = 0.5
    with pytest.raises(evaluator.GenerationEvalError, match="eval_rate_mismatch"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config)
    receipt["observed_counts"]["tool_slots"] = 1199
    with pytest.raises(evaluator.GenerationEvalError, match="eval_receipt_invalid"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config)
    receipt["metrics"]["router"]["eval_proxy"]["confusion_aggregate"][0] -= 1
    with pytest.raises(evaluator.GenerationEvalError, match="eval_count_mismatch"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config)
    receipt["strata"]["tool"]["argument_composition_proxy"]["records"] = 119
    with pytest.raises(evaluator.GenerationEvalError, match="eval_receipt_invalid"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_router_macro_f1_is_recomputed_from_confusion(tmp_path: Path) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    receipt["metrics"]["router"]["eval_proxy"]["confusion_aggregate"] = [
        6,
        1,
        0,
        0,
        7,
        0,
        0,
        0,
        6,
    ]
    with pytest.raises(evaluator.GenerationEvalError, match="eval_rate_mismatch"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_router_macro_f1_zero_denominator_and_quantum_are_exact(
    tmp_path: Path,
) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    router = receipt["metrics"]["router"]["eval_proxy"]
    router["confusion_aggregate"] = [10, 0, 0, 0, 10, 0, 0, 0, 0]
    router["macro_f1"] = "0.666666666667"
    evaluator.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config)
    router = receipt["metrics"]["router"]["eval_proxy"]
    router["confusion_aggregate"] = [10, 0, 0, 0, 10, 0, 0, 0, 0]
    router["macro_f1"] = "0.666666666666"
    with pytest.raises(evaluator.GenerationEvalError, match="eval_rate_mismatch"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config)
    receipt["metrics"]["router"]["eval_proxy"]["macro_f1"] = 1.0
    with pytest.raises(evaluator.GenerationEvalError, match="eval_receipt_invalid"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_every_derived_comparison_is_recomputed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    config, paths = _materialized_case(tmp_path)
    baseline = _passed_receipt(config)
    comparisons = baseline["comparisons"]
    assert isinstance(comparisons, dict)
    assert set(comparisons) == {
        "tool_q_only_minus_base",
        "tool_q_plus_micro_o_minus_q_only",
        "planner_q_only_minus_base",
        "planner_o_only_minus_base",
        "planner_q_plus_micro_o_minus_q_only",
        "identity_correct_minus_base",
        "identity_correct_minus_wrong",
        "wrong_route_macro_degradation",
    }
    for name in comparisons:
        receipt = _passed_receipt(config)
        receipt_comparisons = receipt["comparisons"]
        assert isinstance(receipt_comparisons, dict)
        receipt_comparisons[name] = "0.123456789012"
        with pytest.raises(evaluator.GenerationEvalError, match="eval_rate_mismatch"):
            evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_decimal_comparison_rule_is_deterministic_and_versioned(
    tmp_path: Path,
) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    metrics = receipt["metrics"]
    assert isinstance(metrics, dict)
    metrics["tool"]["tool_q_only"]["execution_projection_exact"] = evaluator.ratio(
        200, 400
    )
    metrics["identity"]["identity_correct_route"]["air_attribution"] = evaluator.ratio(
        25, 50
    )
    metrics["wrong_route"]["derangement_1"]["primary_metric"] = evaluator.ratio(
        430, 860
    )
    receipt["comparisons"] = evaluator.expected_comparisons(metrics)
    comparisons = receipt["comparisons"]
    assert comparisons["tool_q_only_minus_base"] == "-0.500000000000"
    assert comparisons["tool_q_plus_micro_o_minus_q_only"] == "0.500000000000"
    assert comparisons["identity_correct_minus_base"] == "-0.500000000000"
    assert comparisons["identity_correct_minus_wrong"] == "-0.500000000000"
    assert comparisons["wrong_route_macro_degradation"] == "0.166666666667"
    evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_body_token_and_per_record_fields_are_rejected_before_schema(
    tmp_path: Path,
) -> None:
    config, paths = _materialized_case(tmp_path)
    for forbidden_key in (
        "generated_text",
        "token_ids",
        "record_id",
        "per_record",
    ):
        receipt = _passed_receipt(config)
        receipt["metrics"]["tool"]["tool_base"][forbidden_key] = []
        with pytest.raises(
            evaluator.GenerationEvalError,
            match="eval_aggregate_only_violation",
        ):
            evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_kv_claim_cannot_be_upgraded_by_evaluator(tmp_path: Path) -> None:
    config, paths = _materialized_case(tmp_path)
    receipt = _passed_receipt(config)
    receipt["kv_handoff"]["zero_copy"] = True
    with pytest.raises(evaluator.GenerationEvalError, match="eval_receipt_invalid"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config)
    receipt["claims"]["kv_rdma_claimed"] = True
    with pytest.raises(evaluator.GenerationEvalError, match="eval_receipt_invalid"):
        evaluator.validate_receipt(receipt, config, **_dependency_args(paths))


def test_execute_remains_lazy_and_requires_backend(tmp_path: Path) -> None:
    pending = evaluator.load_config()
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_consumer_preflight_pending"
    ):
        evaluator.execute(pending)

    config, paths = _materialized_case(tmp_path)
    with pytest.raises(
        evaluator.GenerationEvalError, match="eval_physical_backend_required"
    ):
        evaluator.execute(
            config,
            **_dependency_args(paths),
        )


def test_cli_redacts_dynamic_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "dynamic-sensitive-marker"

    def reject(_path: str) -> dict[str, object]:
        raise ValueError(marker)

    monkeypatch.setattr(evaluator, "load_config", reject)
    assert evaluator.main(["--status", "--config", "unused"]) == 1
    output = capsys.readouterr().out
    assert marker not in output
    assert json.loads(output)["error_code"] == "runtime_error_redacted"
