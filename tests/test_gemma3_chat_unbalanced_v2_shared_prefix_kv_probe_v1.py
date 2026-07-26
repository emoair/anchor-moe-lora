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
    gemma3_chat_unbalanced_v2_shared_prefix_kv_probe_v1 as probe,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bound_config() -> dict[str, object]:
    config = copy.deepcopy(probe.load_config())
    config.pop("_config_sha256")
    dependency = config["consumer_preflight"]
    assert isinstance(dependency, dict)
    dependency["receipt_sha256"] = _digest("consumer-receipt")
    dependency["artifact_tree_digest_sha256"] = _digest("artifact-tree")
    dependency["shard_inventory_sha256"] = _digest("shard-inventory")
    training = config["training_run"]
    trusted_context = config["trusted_lineage_context"]
    model = config["model"]
    assert isinstance(training, dict)
    assert isinstance(trusted_context, dict)
    assert isinstance(model, dict)
    training["receipt_sha256"] = _digest("training-receipt")
    training["adapter_inventory_sha256"] = _digest("adapter_lineage_sha256")
    training["run_id"] = "diagnostic-run"
    trusted_context["context_sha256"] = _digest("trusted-lineage-context")
    model["model_identity_sha256"] = _digest("model_identity_sha256")
    model["tokenizer_identity_sha256"] = _digest("tokenizer_identity_sha256")
    model["serialization_identity_sha256"] = _digest("serialization_identity_sha256")
    model["template_identity_sha256"] = _digest("template_identity_sha256")
    model["attention_implementation_sha256"] = _digest(
        "attention_implementation_sha256"
    )
    model["dtype_policy_sha256"] = _digest("dtype_policy_sha256")
    model["quantization_policy_sha256"] = _digest("quantization_policy_sha256")
    model["adapter_off_policy_sha256"] = _digest("adapter_off_policy_sha256")
    probe.validate_config(config)
    return config


def _lineage(config: dict[str, object], prefix_length: int = 513) -> dict[str, object]:
    model = config["model"]
    training = config["training_run"]
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    value: dict[str, object] = {field: _digest(field) for field in probe.LINEAGE_FIELDS}
    value.update(
        {
            "model_identity_sha256": model["model_identity_sha256"],
            "tokenizer_identity_sha256": model["tokenizer_identity_sha256"],
            "serialization_identity_sha256": model["serialization_identity_sha256"],
            "template_identity_sha256": model["template_identity_sha256"],
            "attention_implementation_sha256": model["attention_implementation_sha256"],
            "dtype_policy_sha256": model["dtype_policy_sha256"],
            "quantization_policy_sha256": model["quantization_policy_sha256"],
            "adapter_off_policy_sha256": model["adapter_off_policy_sha256"],
            "adapter_lineage_sha256": training["adapter_inventory_sha256"],
            "layer_layout_sha256": model["layer_layout_sha256"],
        }
    )
    value.update(
        {
            "boundary_name": "route_plan_commit",
            "boundary_commit_sha256": _digest("temporary"),
            "adapter_state": "adapter_off_frozen_base",
            "source_cache_present": True,
            "source_cache_complete": True,
            "prefix_length": prefix_length,
        }
    )
    value["boundary_commit_sha256"] = probe.boundary_commit_sha256(value)
    return value


def _dependency_receipts(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    dependency = config["consumer_preflight"]
    training = config["training_run"]
    model = config["model"]
    assert isinstance(dependency, dict)
    assert isinstance(training, dict)
    assert isinstance(model, dict)
    consumer_receipt = {
        "schema_version": probe.CONSUMER_RECEIPT_VERSION,
        "status": "passed",
        "identity": {
            "artifact_tree_digest_sha256": dependency["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": dependency["shard_inventory_sha256"],
            "model_identity_sha256": model["model_identity_sha256"],
            "tokenizer_identity_sha256": model["tokenizer_identity_sha256"],
            "serialization_identity_sha256": model["serialization_identity_sha256"],
        },
    }
    training_receipt = {
        "schema_version": probe.TRAINING_RECEIPT_VERSION,
        "status": "passed_diagnostic_multiarm_training",
        "run_id": training["run_id"],
        "identity": {
            "consumer_preflight_receipt_sha256": dependency["receipt_sha256"],
            "artifact_tree_digest_sha256": dependency["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": dependency["shard_inventory_sha256"],
            "adapter_inventory_sha256": training["adapter_inventory_sha256"],
        },
        "claims": {
            "fresh_base": True,
            "fresh_adapter": True,
            "resume": False,
        },
    }
    return consumer_receipt, training_receipt


ReceiptMutator = Callable[[dict[str, object]], None]


def _trusted_context(
    config: dict[str, object], prefix_length: int = 513
) -> dict[str, object]:
    dependency = config["consumer_preflight"]
    training = config["training_run"]
    assert isinstance(dependency, dict)
    assert isinstance(training, dict)
    return {
        "schema_version": probe.TRUSTED_CONTEXT_VERSION,
        "status": probe.TRUSTED_CONTEXT_STATUS,
        "identity": {
            "consumer_preflight_receipt_sha256": dependency["receipt_sha256"],
            "artifact_tree_digest_sha256": dependency["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": dependency["shard_inventory_sha256"],
            "training_run_receipt_sha256": training["receipt_sha256"],
            "adapter_inventory_sha256": training["adapter_inventory_sha256"],
            "run_id": training["run_id"],
        },
        "lineage": _lineage(config, prefix_length),
        "claims": {
            "external_to_experiments": True,
            "body_free": True,
            "adapter_off": True,
        },
    }


def _write_json(path: Path, value: dict[str, object]) -> str:
    raw = (
        json.dumps(
            value,
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


def _physical_inputs(
    tmp_path: Path,
    config: dict[str, object],
    *,
    prefix_length: int = 513,
    mutate_consumer: ReceiptMutator | None = None,
    mutate_consumer_schema: ReceiptMutator | None = None,
    mutate_training: ReceiptMutator | None = None,
    mutate_training_schema: ReceiptMutator | None = None,
    mutate_context: ReceiptMutator | None = None,
    mutate_context_schema: ReceiptMutator | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    dependency = config["consumer_preflight"]
    training_config = config["training_run"]
    context_config = config["trusted_lineage_context"]
    assert isinstance(dependency, dict)
    assert isinstance(training_config, dict)
    assert isinstance(context_config, dict)

    consumer, _ = _dependency_receipts(config)
    if mutate_consumer is not None:
        mutate_consumer(consumer)
    consumer_path = tmp_path / "consumer-receipt.json"
    dependency["receipt_sha256"] = _write_json(consumer_path, consumer)
    consumer_schema = _closed_schema(consumer)
    if mutate_consumer_schema is not None:
        mutate_consumer_schema(consumer_schema)
    consumer_schema_path = tmp_path / "consumer-receipt.schema.json"
    dependency["receipt_schema_sha256"] = _write_json(
        consumer_schema_path, consumer_schema
    )

    _, training = _dependency_receipts(config)
    if mutate_training is not None:
        mutate_training(training)
    training_path = tmp_path / "training-receipt.json"
    training_config["receipt_sha256"] = _write_json(training_path, training)
    training_schema = _closed_schema(training)
    if mutate_training_schema is not None:
        mutate_training_schema(training_schema)
    training_schema_path = tmp_path / "training-receipt.schema.json"
    training_config["receipt_schema_sha256"] = _write_json(
        training_schema_path, training_schema
    )

    context = _trusted_context(config, prefix_length)
    if mutate_context is not None:
        mutate_context(context)
    context_path = tmp_path / "trusted-lineage-context.json"
    context_config["context_sha256"] = _write_json(context_path, context)
    context_schema = _closed_schema(context)
    if mutate_context_schema is not None:
        mutate_context_schema(context_schema)
    context_schema_path = tmp_path / "trusted-lineage-context.schema.json"
    context_config["context_schema_sha256"] = _write_json(
        context_schema_path, context_schema
    )
    probe.validate_config(config)
    return (
        consumer_path,
        consumer_schema_path,
        training_path,
        training_schema_path,
        context_path,
        context_schema_path,
    )


def _dependency_args(
    paths: tuple[Path, Path, Path, Path, Path, Path],
) -> dict[str, Path]:
    return {
        "consumer_receipt_path": paths[0],
        "consumer_receipt_schema_path": paths[1],
        "training_receipt_path": paths[2],
        "training_receipt_schema_path": paths[3],
        "trusted_lineage_context_path": paths[4],
        "trusted_lineage_context_schema_path": paths[5],
    }


def _materialized_case(
    tmp_path: Path, prefix_length: int = 513
) -> tuple[
    dict[str, object],
    tuple[Path, Path, Path, Path, Path, Path],
    dict[str, object],
]:
    config = _bound_config()
    paths = _physical_inputs(tmp_path, config, prefix_length=prefix_length)
    trusted_context = json.loads(paths[4].read_text(encoding="utf-8"))
    assert isinstance(trusted_context, dict)
    return config, paths, trusted_context


def _passed_receipt(
    config: dict[str, object],
    trusted_context: dict[str, object],
    prefix_length: int = 513,
) -> dict[str, object]:
    dependency = config["consumer_preflight"]
    training = config["training_run"]
    model = config["model"]
    assert isinstance(dependency, dict)
    assert isinstance(training, dict)
    assert isinstance(model, dict)
    layers = []
    for layout in probe.layer_layout():
        index = int(layout["layer_index"])
        start, end = probe.retained_span(index, prefix_length)
        digest = _digest(f"layer-{index}-retained-values")
        layers.append(
            {
                **layout,
                "retained_start": start,
                "retained_end": end,
                "retained_length": end - start,
                "isolated_value_sha256": digest,
                "handoff_value_sha256": digest,
                "value_equal": True,
                "pointer_equal_observed": True,
                "storage_shared_claimed": False,
            }
        )
    lineage = copy.deepcopy(trusted_context["lineage"])
    assert isinstance(lineage, dict)
    prefix_digest = str(lineage["ordered_prefix_sha256"])
    return {
        "schema_version": probe.RECEIPT_VERSION,
        "profile_id": probe.PROFILE_ID,
        "status": "passed_diagnostic_prefix_value_handoff",
        "identity": {
            "config_sha256": probe.config_sha256(config),
            "implementation_sha256": probe.implementation_sha256(),
            "consumer_preflight_receipt_sha256": dependency["receipt_sha256"],
            "artifact_tree_digest_sha256": dependency["artifact_tree_digest_sha256"],
            "shard_inventory_sha256": dependency["shard_inventory_sha256"],
            "training_run_receipt_sha256": training["receipt_sha256"],
            "adapter_inventory_sha256": training["adapter_inventory_sha256"],
            "run_id": training["run_id"],
            "trusted_lineage_context_sha256": config["trusted_lineage_context"][
                "context_sha256"
            ],
            "model_identity_sha256": model["model_identity_sha256"],
            "tokenizer_identity_sha256": model["tokenizer_identity_sha256"],
            "serialization_identity_sha256": model["serialization_identity_sha256"],
            "template_identity_sha256": model["template_identity_sha256"],
            "attention_implementation_sha256": model["attention_implementation_sha256"],
            "dtype_policy_sha256": model["dtype_policy_sha256"],
            "quantization_policy_sha256": model["quantization_policy_sha256"],
            "adapter_off_policy_sha256": model["adapter_off_policy_sha256"],
            "layer_layout_sha256": probe.layer_layout_sha256(),
        },
        "boundary": {
            "name": "route_plan_commit",
            "adapter_state": "adapter_off_frozen_base",
            "commit_sha256": lineage["boundary_commit_sha256"],
            "route_commit_sha256": lineage["route_commit_sha256"],
            "plan_commit_sha256": lineage["plan_commit_sha256"],
            "ordered_prefix_sha256": lineage["ordered_prefix_sha256"],
            "prefix_length": prefix_length,
            "retained_span_algorithm": (
                "full_0_to_boundary__sliding_max_0_boundary_minus_512_to_boundary"
            ),
            "source_cache_present": True,
            "source_cache_complete": True,
        },
        "experiments": {
            "isolated_reprefill": {
                "executed": True,
                "prefill_count": 2,
                "transport": "none_isolated_reprefill",
                "source_prefix_sha256": prefix_digest,
                "result_digest_sha256": _digest("isolated-result"),
                "first_token_distribution_digest_sha256": _digest(
                    "first-token-distribution"
                ),
                "first_token_argmax_digest_sha256": _digest("first-token-argmax"),
                "lineage": copy.deepcopy(lineage),
            },
            "one_prefill_explicit_local_copy": {
                "executed": True,
                "prefill_count": 1,
                "transport": "explicit_local_copy",
                "source_prefix_sha256": prefix_digest,
                "result_digest_sha256": _digest("handoff-result"),
                "first_token_distribution_digest_sha256": _digest(
                    "first-token-distribution"
                ),
                "first_token_argmax_digest_sha256": _digest("first-token-argmax"),
                "lineage": copy.deepcopy(lineage),
            },
        },
        "layers": layers,
        "diagnostics": {
            "all_layer_retained_values_equal": True,
            "first_token_distribution_digest_equal": True,
            "first_token_argmax_equal": True,
            "source_cache_unchanged": True,
            "cross_mutation_rejections": {
                "attempted": len(probe.REQUIRED_MUTATIONS),
                "rejected": len(probe.REQUIRED_MUTATIONS),
            },
            "cache_miss_rejected": True,
            "private_tail_append_only": True,
            "private_tails_alias": False,
            "cache_object_equal_observed": True,
            "cache_pointer_equal_observed": True,
        },
        "performance": {
            "isolated_prefill_ms": 8.0,
            "handoff_prefill_ms": 4.0,
            "handoff_copy_ms": 0.5,
            "first_token_ms": 1.0,
            "peak_device_bytes": 1024,
            "repetitions": 3,
        },
        "failure_codes": [],
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "exact_prefix_value_handoff": True,
            "prefill_compute_handoff": True,
            "shared_storage": False,
            "zero_copy": False,
            "full_generation_kv_shared": False,
            "rdma": False,
            "physical_execution_completed": True,
            "formal": False,
        },
        "audit": {
            "dataset_body_reads": 0,
            "raw_token_ids_persisted": 0,
            "sample_bodies_persisted": 0,
            "gold_reads": 0,
            "heldout_body_reads": 0,
            "protected_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
        },
    }


def test_checked_in_config_and_schemas_are_valid_and_model_free() -> None:
    config = probe.load_config()
    assert probe.build_status(config)["status"] == (
        "blocked_waiting_for_authenticated_identities"
    )
    assert probe.build_status(config)["audit"]["gpu_requests"] == 0
    assert tuple(config["consumer_preflight"].values()).count("pending") == 4
    assert config["consumer_preflight"]["required_schema_version"].endswith(
        ".sharded-v1"
    )
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    for path in (probe.CONFIG_SCHEMA_PATH, probe.RECEIPT_SCHEMA_PATH):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_layer_layout_is_22_sliding_plus_4_full_and_digest_bound() -> None:
    layout = probe.layer_layout()
    assert len(layout) == 26
    assert sum(item["attention_kind"] == "sliding" for item in layout) == 22
    assert sum(item["attention_kind"] == "full" for item in layout) == 4
    assert tuple(
        item["layer_index"] for item in layout if item["attention_kind"] == "full"
    ) == (5, 11, 17, 23)
    config = probe.load_config()
    assert config["model"]["layer_layout_sha256"] == probe.layer_layout_sha256()


@pytest.mark.parametrize(
    ("prefix_length", "sliding_span", "full_span"),
    [
        (127, (0, 127), (0, 127)),
        (511, (0, 511), (0, 511)),
        (512, (0, 512), (0, 512)),
        (513, (1, 513), (0, 513)),
        (1025, (513, 1025), (0, 1025)),
    ],
)
def test_retained_span_covers_below_at_and_above_window(
    prefix_length: int,
    sliding_span: tuple[int, int],
    full_span: tuple[int, int],
) -> None:
    assert probe.retained_span(0, prefix_length) == sliding_span
    assert probe.retained_span(5, prefix_length) == full_span


def test_complete_lineage_and_boundary_commit_pass() -> None:
    source = _lineage(_bound_config())
    handoff = copy.deepcopy(source)
    assert (
        probe.validate_handoff_lineage(source, handoff)
        == source["boundary_commit_sha256"]
    )


@pytest.mark.parametrize("field", probe.LINEAGE_FIELDS)
def test_every_lineage_hash_mutation_fails_closed(field: str) -> None:
    source = _lineage(_bound_config())
    handoff = copy.deepcopy(source)
    handoff[field] = _digest(f"mutated-{field}")
    handoff["boundary_commit_sha256"] = probe.boundary_commit_sha256(handoff)
    with pytest.raises(probe.SharedPrefixKVError, match="kv_lineage_mismatch"):
        probe.validate_handoff_lineage(source, handoff)


def test_adapter_off_boundary_and_cache_miss_mutations_fail_closed() -> None:
    source = _lineage(_bound_config())
    handoff = copy.deepcopy(source)
    handoff["adapter_state"] = "expert_adapter_on"
    with pytest.raises(probe.SharedPrefixKVError, match="kv_adapter_state_invalid"):
        probe.validate_handoff_lineage(source, handoff)

    handoff = copy.deepcopy(source)
    handoff["source_cache_present"] = False
    with pytest.raises(probe.SharedPrefixKVError, match="kv_cache_miss"):
        probe.validate_handoff_lineage(source, handoff)

    handoff = copy.deepcopy(source)
    handoff["boundary_commit_sha256"] = _digest("wrong-boundary")
    with pytest.raises(probe.SharedPrefixKVError, match="kv_boundary_commit_mismatch"):
        probe.validate_handoff_lineage(source, handoff)


def test_old_unsharded_consumer_receipt_version_is_rejected(tmp_path: Path) -> None:
    config = _bound_config()

    def old_version(value: dict[str, object]) -> None:
        value["schema_version"] = (
            "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.v1"
        )

    paths = _physical_inputs(
        tmp_path,
        config,
        mutate_consumer=old_version,
    )
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.build_preflight(
            config,
            **_dependency_args(paths),
        )


@pytest.mark.parametrize(
    ("target", "error_code"),
    [
        ("consumer", "kv_consumer_preflight_pending"),
        ("training", "kv_training_run_pending"),
        ("context", "kv_trusted_lineage_context_pending"),
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
def test_physical_dependency_closed_contract_rejects_shape_drift(
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
    paths = _physical_inputs(tmp_path, config, **kwargs)
    with pytest.raises(probe.SharedPrefixKVError, match=error_code):
        probe.build_preflight(
            config,
            **_dependency_args(paths),
        )


def test_physical_dependency_sha_and_terminal_recheck_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _ = _materialized_case(tmp_path / "sha")
    paths[0].write_bytes(b"{}\n")
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.build_preflight(
            config,
            **_dependency_args(paths),
        )

    config, paths, _ = _materialized_case(tmp_path / "terminal")
    original = probe._stat_signature
    calls = 0

    def drift(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = original(value)
        if calls == 4:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(probe, "_stat_signature", drift)
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.build_preflight(
            config,
            **_dependency_args(paths),
        )


@pytest.mark.parametrize(
    ("missing_index", "error_code"),
    [
        (0, "kv_consumer_preflight_pending"),
        (1, "kv_consumer_preflight_pending"),
        (2, "kv_training_run_pending"),
        (3, "kv_training_run_pending"),
        (4, "kv_trusted_lineage_context_pending"),
        (5, "kv_trusted_lineage_context_pending"),
    ],
)
def test_every_dependency_receipt_and_schema_is_mandatory(
    tmp_path: Path, missing_index: int, error_code: str
) -> None:
    config, paths, _ = _materialized_case(tmp_path)
    arguments = _dependency_args(paths)
    arguments[tuple(arguments)[missing_index]] = None
    with pytest.raises(probe.SharedPrefixKVError, match=error_code):
        probe.build_preflight(config, **arguments)


@pytest.mark.parametrize("target", ("consumer", "training", "context"))
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

    paths = _physical_inputs(
        tmp_path,
        config,
        **{f"mutate_{target}_schema": mutate},
    )
    error_code = {
        "consumer": "kv_consumer_preflight_pending",
        "training": "kv_training_run_pending",
        "context": "kv_trusted_lineage_context_pending",
    }[target]
    with pytest.raises(probe.SharedPrefixKVError, match=error_code):
        probe.build_preflight(config, **_dependency_args(paths))


def test_dependency_schema_sha_and_terminal_recheck_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _ = _materialized_case(tmp_path / "sha")
    paths[1].write_bytes(b"{}\n")
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.build_preflight(config, **_dependency_args(paths))

    config, paths, _ = _materialized_case(tmp_path / "terminal")
    original = probe._stat_signature
    calls = 0

    def drift(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = original(value)
        if calls == 8:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(probe, "_stat_signature", drift)
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.build_preflight(config, **_dependency_args(paths))


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
    config, paths, _ = _materialized_case(tmp_path)
    paths[path_index].write_bytes(raw)
    dependency = config["consumer_preflight"]
    assert isinstance(dependency, dict)
    field = "receipt_sha256" if path_index == 0 else "receipt_schema_sha256"
    dependency[field] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.build_preflight(config, **_dependency_args(paths))


def test_public_receipt_validation_cannot_bypass_dependencies(tmp_path: Path) -> None:
    config, _, trusted_context = _materialized_case(tmp_path)
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.validate_receipt(_passed_receipt(config, trusted_context), config)


def test_passed_receipt_accepts_pointer_equality_without_storage_claim(
    tmp_path: Path,
) -> None:
    config, paths, trusted_context = _materialized_case(tmp_path)
    receipt = _passed_receipt(config, trusted_context)
    validated = probe.validate_receipt(
        receipt,
        config,
        **_dependency_args(paths),
    )
    assert validated["status"] == "passed_diagnostic_prefix_value_handoff"
    assert all(layer["pointer_equal_observed"] for layer in validated["layers"])
    assert all(
        layer["storage_shared_claimed"] is False for layer in validated["layers"]
    )
    assert validated["claims"]["shared_storage"] is False
    assert validated["claims"]["zero_copy"] is False
    assert validated["claims"]["rdma"] is False


@pytest.mark.parametrize(
    ("operand", "diagnostic"),
    [
        (
            "first_token_distribution_digest_sha256",
            "first_token_distribution_digest_equal",
        ),
        ("first_token_argmax_digest_sha256", "first_token_argmax_equal"),
    ],
)
def test_first_token_equivalence_is_derived_from_physical_digest_operands(
    tmp_path: Path, operand: str, diagnostic: str
) -> None:
    config, paths, trusted_context = _materialized_case(tmp_path)
    receipt = _passed_receipt(config, trusted_context)
    receipt["experiments"]["one_prefill_explicit_local_copy"][operand] = _digest(
        f"drifted-{operand}"
    )
    assert receipt["diagnostics"][diagnostic] is True
    with pytest.raises(probe.SharedPrefixKVError, match="kv_first_token_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["diagnostics"][diagnostic] = False
    with pytest.raises(probe.SharedPrefixKVError, match="kv_first_token_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))


def test_retained_value_source_tail_and_claim_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    config, paths, trusted_context = _materialized_case(tmp_path)

    receipt = _passed_receipt(config, trusted_context)
    receipt["layers"][0]["retained_start"] += 1
    with pytest.raises(probe.SharedPrefixKVError, match="kv_retained_span_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["layers"][0]["handoff_value_sha256"] = _digest("wrong-value")
    receipt["layers"][0]["value_equal"] = False
    with pytest.raises(probe.SharedPrefixKVError, match="kv_layer_value_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["diagnostics"]["source_cache_unchanged"] = False
    with pytest.raises(probe.SharedPrefixKVError, match="kv_source_cache_mutated"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["diagnostics"]["private_tails_alias"] = True
    with pytest.raises(probe.SharedPrefixKVError, match="kv_private_tail_alias"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["claims"]["shared_storage"] = True
    with pytest.raises(probe.SharedPrefixKVError, match="kv_receipt_invalid"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))


def test_pass_requires_executed_lineage_prefix_and_complete_mutation_evidence(
    tmp_path: Path,
) -> None:
    config, paths, trusted_context = _materialized_case(tmp_path)

    receipt = _passed_receipt(config, trusted_context)
    receipt["experiments"]["isolated_reprefill"]["executed"] = False
    with pytest.raises(probe.SharedPrefixKVError, match="kv_receipt_invalid"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["experiments"]["isolated_reprefill"]["lineage"][
        "boundary_commit_sha256"
    ] = _digest("replaced-boundary")
    with pytest.raises(probe.SharedPrefixKVError, match="kv_boundary_commit_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    handoff_lineage = receipt["experiments"]["one_prefill_explicit_local_copy"][
        "lineage"
    ]
    handoff_lineage["ordered_prefix_sha256"] = _digest("different-prefix")
    handoff_lineage["boundary_commit_sha256"] = probe.boundary_commit_sha256(
        handoff_lineage
    )
    receipt["experiments"]["one_prefill_explicit_local_copy"][
        "source_prefix_sha256"
    ] = handoff_lineage["ordered_prefix_sha256"]
    with pytest.raises(probe.SharedPrefixKVError, match="kv_lineage_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["diagnostics"]["cross_mutation_rejections"] = {
        "attempted": len(probe.REQUIRED_MUTATIONS) - 1,
        "rejected": len(probe.REQUIRED_MUTATIONS) - 1,
    }
    with pytest.raises(probe.SharedPrefixKVError, match="kv_receipt_invalid"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))

    receipt = _passed_receipt(config, trusted_context)
    receipt["diagnostics"]["cache_miss_rejected"] = False
    with pytest.raises(probe.SharedPrefixKVError, match="kv_receipt_invalid"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))


def test_synchronized_two_arm_lineage_drift_with_recomputed_commit_is_rejected(
    tmp_path: Path,
) -> None:
    config, paths, trusted_context = _materialized_case(tmp_path)
    receipt = _passed_receipt(config, trusted_context)
    for experiment in receipt["experiments"].values():
        lineage = experiment["lineage"]
        lineage["template_identity_sha256"] = _digest("synchronized-drift")
        lineage["boundary_commit_sha256"] = probe.boundary_commit_sha256(lineage)
    drifted = receipt["experiments"]["isolated_reprefill"]["lineage"]
    receipt["boundary"]["commit_sha256"] = drifted["boundary_commit_sha256"]
    with pytest.raises(probe.SharedPrefixKVError, match="kv_lineage_mismatch"):
        probe.validate_receipt(receipt, config, **_dependency_args(paths))


def test_execute_is_lazy_and_requires_consumer_then_backend(tmp_path: Path) -> None:
    pending = probe.load_config()
    with pytest.raises(
        probe.SharedPrefixKVError, match="kv_consumer_preflight_pending"
    ):
        probe.execute(pending)

    config, paths, _ = _materialized_case(tmp_path)
    with pytest.raises(probe.SharedPrefixKVError, match="kv_physical_backend_required"):
        probe.execute(
            config,
            **_dependency_args(paths),
        )


def test_cli_redacts_arbitrary_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "sensitive-dynamic-marker"

    def reject(_path: str) -> dict[str, object]:
        raise ValueError(marker)

    monkeypatch.setattr(probe, "load_config", reject)
    assert probe.main(["--status", "--config", "unused"]) == 1
    output = capsys.readouterr().out
    assert marker not in output
    assert json.loads(output)["error_code"] == "runtime_error_redacted"
