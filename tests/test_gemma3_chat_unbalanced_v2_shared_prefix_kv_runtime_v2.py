from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.research import (
    gemma3_chat_unbalanced_v2_shared_prefix_kv_runtime_v2 as runtime,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _ready_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    config = copy.deepcopy(runtime.load_config())
    config.pop("_config_sha256", None)
    config["execution"]["execute_authorized"] = True
    config["model"]["tokenizer_identity_sha256"] = _sha("tokenizer")
    config["teacher_final"] = {
        "schema_version": "unit.teacher-final.v2",
        "binding_sha256": _sha("teacher-binding"),
        "manifest_sha256": _sha("teacher-manifest"),
        "record_schema_sha256": _sha("teacher-record-schema"),
        "release_receipt_sha256": _sha("teacher-release-receipt"),
        "release_attestation_sha256": _sha("teacher-release-attestation"),
        "shard_inventory_sha256": _sha("teacher-shard-inventory"),
        "public_identity_sha256": _sha("teacher-public-identity"),
    }
    model_root = tmp_path / "model"
    model_root.mkdir()
    base_files: dict[str, str] = {}
    for name, raw in {
        "config.json": b'{"model_type":"gemma3_text"}\n',
        "model.safetensors": b"synthetic-model-weights\n",
    }.items():
        (model_root / name).write_bytes(raw)
        base_files[name] = hashlib.sha256(raw).hexdigest()
    base_inventory = hashlib.sha256(_canonical(base_files)).hexdigest()
    config["model"]["model_identity_sha256"] = base_inventory
    adapters: dict[str, Any] = {}
    for arm in runtime.TRAINED_ARMS:
        adapter_root = tmp_path / "adapters" / arm
        adapter_root.mkdir(parents=True)
        entries: dict[str, Any] = {}
        for key, filename, raw in (
            (
                "adapter_config",
                "adapter_config.json",
                _canonical({"arm": arm}),
            ),
            (
                "adapter_model",
                "adapter_model.safetensors",
                f"synthetic:{arm}\n".encode("ascii"),
            ),
        ):
            (adapter_root / filename).write_bytes(raw)
            entries[key] = {
                "path": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        tree = hashlib.sha256(
            _canonical(
                sorted(
                    entries.values(),
                    key=lambda item: item["path"],
                )
            )
        ).hexdigest()
        adapters[arm] = {
            "path": f"adapters/{arm}",
            "adapter_profile": runtime.ARM_PROFILES[arm],
            "trainable_parameters": 1,
            **entries,
            "artifact_tree_sha256": tree,
            "tensor_inventory_sha256": _sha(f"inventory:{arm}"),
        }
    phase_base = {arm: _sha("base-parameters") for arm in runtime.TRAINED_ARMS}
    phase_scb = {arm: _sha("q8-scb") for arm in runtime.TRAINED_ARMS}
    adapter_receipt = {
        "schema_version": config["adapter_receipt"]["schema_version"],
        "status": config["adapter_receipt"]["required_status"],
        "run_id": "unit-test",
        "mode": "full",
        "config": {},
        "source": {
            "consumer_preflight_receipt_sha256": config["consumer_preflight"][
                "receipt_sha256"
            ]
        },
        "teacher_final": {
            **config["teacher_final"],
            **runtime.TEACHER_FINAL_METRICS,
            "producer_original_target_fallback": False,
        },
        "tokenizer": {
            "combined_identity_sha256": config["model"]["tokenizer_identity_sha256"]
        },
        "model": {
            "base_files": base_files,
            "base_file_inventory_sha256": base_inventory,
            "phase_base_hashes": phase_base,
            "phase_q8_scb_hashes": phase_scb,
            "all_phase_base_hashes_equal": True,
            "all_phase_q8_scb_hashes_equal": True,
            "frozen_q8_base": True,
            "base_o_proj_always_frozen": True,
        },
        "phase_order": list(runtime.TRAINED_ARMS),
        "phase_receipts": [
            {
                "arm": arm,
                "phase": "full",
                "status": "passed",
                "base_parameter_sha256": phase_base[arm],
                "q8_scb_sha256": phase_scb[arm],
            }
            for arm in runtime.TRAINED_ARMS
        ],
        "adapters": adapters,
        "training_lineage": {},
        "launcher": {},
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "release": False,
            "sample_bodies_included": False,
            "raw_token_ids_included": False,
            "producer_original_targets_used": False,
            "physical_kv_validated": False,
            "full_generation_kv_shared": False,
        },
    }
    adapter_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "schema_version",
            "status",
            "run_id",
            "mode",
            "config",
            "source",
            "teacher_final",
            "tokenizer",
            "model",
            "phase_order",
            "phase_receipts",
            "adapters",
            "training_lineage",
            "launcher",
            "claims",
        ],
    }
    receipt_path = tmp_path / "adapter.json"
    sidecar_path = tmp_path / "adapter.json.sha256"
    schema_path = tmp_path / "adapter.schema.json"
    receipt_raw = _canonical(adapter_receipt)
    schema_raw = _canonical(adapter_schema)
    receipt_path.write_bytes(receipt_raw)
    sidecar_raw = (
        f"{hashlib.sha256(receipt_raw).hexdigest()}  {receipt_path.name}\n".encode(
            "ascii"
        )
    )
    sidecar_path.write_bytes(sidecar_raw)
    schema_path.write_bytes(schema_raw)
    config["adapter_receipt"].update(
        {
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_raw).hexdigest(),
        }
    )
    runtime.validate_config(config)
    return (
        config,
        {
            "receipt": receipt_path,
            "sidecar": sidecar_path,
            "schema": schema_path,
        },
        adapter_receipt,
    )


def _serialized_prefix() -> str:
    return (
        "<start_of_turn>user\n"
        "system contract\n\nuser request"
        "<end_of_turn><eos>\n<start_of_turn>model\n"
    )


def _source_identity(config: dict[str, Any], source_blob: str) -> dict[str, Any]:
    producer = config["producer_final"]
    prefix = _serialized_prefix()
    return {
        "producer_candidate_commit": producer["candidate_commit"],
        "producer_release_commit": producer["release_commit"],
        "producer_release_tree": producer["release_tree"],
        "source_blob_path": source_blob,
        "source_blob_sha256": _sha("source-blob"),
        "source_blob_bytes": 1234,
        "record_index": 7,
        "record_sha256": _sha("record"),
        "record_prompt_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "example_serialization_sha256": _sha("example-serialization"),
        "tokenizer_binding_sha256": runtime.tokenizer_binding_sha256(config),
        "model_config_sha256": _sha("model-config"),
        "training_serialization_inventory_sha256": producer[
            "serialization_inventory_sha256"
        ],
        "route_commit_sha256": _sha("route"),
        "plan_commit_sha256": _sha("plan"),
    }


def _source_gate(config: dict[str, Any], source_blob: str):
    identity = _source_identity(config, source_blob)

    def gate(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["source_blob_path"] == source_blob
        assert kwargs["record_index"] == 7
        assert "current_head" not in kwargs["expected"]
        assert "upstream" not in kwargs["expected"]
        return {
            "prefix_input_ids": (2, 105, 2364, 107, 106, 1, 107, 105, 4368, 107),
            "source_identity": identity,
        }

    return gate


def _model_config() -> dict[str, Any]:
    full = {5, 11, 17, 23}
    return {
        "num_hidden_layers": 26,
        "sliding_window": 512,
        "layer_types": [
            "full_attention" if index in full else "sliding_attention"
            for index in range(26)
        ],
    }


def _backend(
    config: dict[str, Any],
    source_identity: dict[str, Any],
    *,
    boundary_drift: bool = False,
    private_alias: bool = False,
    source_mutated: bool = False,
    prefix_mismatch: bool = False,
    output_prefix_mismatch: bool = False,
):
    topology = runtime.derive_layer_topology(_model_config(), config["model"])
    token_order = _sha("token-order")
    position = _sha("positions")
    mask = _sha("mask")
    rope = _sha("rope")

    def run(request: dict[str, Any]) -> dict[str, Any]:
        lineage = {
            "producer_candidate_commit": source_identity["producer_candidate_commit"],
            "producer_release_commit": source_identity["producer_release_commit"],
            "producer_release_tree": source_identity["producer_release_tree"],
            "source_blob_sha256": source_identity["source_blob_sha256"],
            "source_record_sha256": source_identity["record_sha256"],
            "training_serialization_inventory_sha256": source_identity[
                "training_serialization_inventory_sha256"
            ],
            "record_prompt_sha256": source_identity["record_prompt_sha256"],
            "example_serialization_sha256": source_identity[
                "example_serialization_sha256"
            ],
            "tokenizer_binding_sha256": source_identity["tokenizer_binding_sha256"],
            "model_config_sha256": source_identity["model_config_sha256"],
            "token_order_sha256": token_order,
            "position_sha256": position,
            "attention_mask_sha256": mask,
            "rope_sha256": rope,
            "model_identity_sha256": config["model"]["model_identity_sha256"],
            "base_identity_sha256": config["model"]["model_identity_sha256"],
            "q8_scb_sha256": request["q8_scb_sha256"],
            "adapter_lineage_sha256": request["adapter_inventory_sha256"],
            "layer_topology_sha256": topology["layer_topology_sha256"],
            "route_commit_sha256": source_identity["route_commit_sha256"],
            "plan_commit_sha256": source_identity["plan_commit_sha256"],
        }
        commit = runtime.boundary_commit_sha256(lineage)
        if boundary_drift:
            commit = _sha("wrong-boundary")
        layers = []
        for detail in topology["layers_detail"]:
            capacity = runtime._cache_capacity(detail)
            source_tokens = len(request["prefix_input_ids"])
            if capacity is not None:
                source_tokens = min(source_tokens, capacity)
            planner_tokens, planner_retained = runtime._retained_prefix_layout(
                detail,
                source_tokens=source_tokens,
                tail_tokens=1,
            )
            output_tokens, output_retained = runtime._retained_prefix_layout(
                detail,
                source_tokens=source_tokens,
                tail_tokens=2,
            )
            cold_key = _sha(f"cold-key-{detail['index']}")
            cold_value = _sha(f"cold-value-{detail['index']}")
            planner_key = _sha(f"planner-key-{detail['index']}")
            planner_value = _sha(f"planner-value-{detail['index']}")
            output_key = _sha(f"output-key-{detail['index']}")
            output_value = _sha(f"output-value-{detail['index']}")
            layers.append(
                {
                    **detail,
                    "sequence_axis": -2,
                    "window_semantics": (
                        "full_prefix_unbounded"
                        if capacity is None
                        else "retained_suffix_window_minus_one"
                    ),
                    "cache_capacity_tokens": capacity,
                    "source_cache_tokens": source_tokens,
                    "planner_cache_tokens": planner_tokens,
                    "output_cache_tokens": output_tokens,
                    "cold_prefill_key_sha256": cold_key,
                    "cold_prefill_value_sha256": cold_value,
                    "source_prefill_key_sha256": cold_key,
                    "source_prefill_value_sha256": cold_value,
                    "prefill_recompute_value_equal": True,
                    "planner_retained_prefix_tokens": planner_retained,
                    "planner_source_prefix_key_sha256": planner_key,
                    "planner_source_prefix_value_sha256": planner_value,
                    "planner_branch_prefix_key_sha256": planner_key,
                    "planner_branch_prefix_value_sha256": (
                        _sha("mismatched-layer")
                        if prefix_mismatch and detail["index"] == 0
                        else planner_value
                    ),
                    "planner_prefix_value_equal": not (
                        prefix_mismatch and detail["index"] == 0
                    ),
                    "output_retained_prefix_tokens": output_retained,
                    "output_source_prefix_key_sha256": output_key,
                    "output_source_prefix_value_sha256": output_value,
                    "output_branch_prefix_key_sha256": output_key,
                    "output_branch_prefix_value_sha256": (
                        _sha("mismatched-output-layer")
                        if output_prefix_mismatch and detail["index"] == 0
                        else output_value
                    ),
                    "output_prefix_value_equal": not (
                        output_prefix_mismatch and detail["index"] == 0
                    ),
                    "source_to_planner_pointer_equal_observed": False,
                    "source_to_output_pointer_equal_observed": False,
                    "storage_pointer_is_proof": False,
                }
            )
        logits = _sha("logits")
        next_token = _sha("next-token")
        return {
            "topology": {**topology, "layers_detail": layers},
            "boundary": {
                "boundary_commit_sha256": commit,
                "ordered_prefix_sha256": source_identity["record_prompt_sha256"],
                "token_order_sha256": token_order,
                "position_sha256": position,
                "attention_mask_sha256": mask,
                "rope_sha256": rope,
                "route_commit_sha256": source_identity["route_commit_sha256"],
                "plan_commit_sha256": source_identity["plan_commit_sha256"],
                "prefix_tokens": len(request["prefix_input_ids"]),
                "source_prefill_adapter_state": "adapter_off_frozen_base",
                "planner_adapter_sha256": request["planner_adapter_sha256"],
                "output_adapter_sha256": request["output_adapter_sha256"],
                "latent_packet": {
                    "enabled": False,
                    "fixed_dimension": 0,
                    "serialization_sha256": None,
                    "position_sha256": None,
                    "commit_sha256": None,
                    "hidden_chain_of_thought": False,
                },
            },
            "comparison": {
                "cold_recompute_executed": True,
                "shared_prefix_handoff_executed": True,
                "logits_digest_cold": logits,
                "logits_digest_handoff": logits,
                "logits_allclose": True,
                "logits_max_abs_error": 0.0,
                "next_token_argmax_equal": True,
                "next_token_digest_cold": next_token,
                "next_token_digest_handoff": next_token,
                "prefill_count_cold": 1,
                "prefill_count_handoff": 1,
            },
            "isolation": {
                "source_cache_unchanged": not source_mutated,
                "planner_tail_private": True,
                "output_tail_private": True,
                "sibling_branch_unchanged": True,
                "private_tail_alias": private_alias,
                "wrong_adapter_rejected": True,
                "wrong_position_rejected": True,
                "wrong_attention_mask_rejected": True,
                "cache_mutation_rejected": not source_mutated,
                "pointer_observation_only": True,
            },
            "performance": {
                "cold_latency_ms": 5.0,
                "handoff_latency_ms": 3.0,
                "handoff_prefill_ms": 1.0,
                "planner_tail_ms": 1.0,
                "output_tail_ms": 1.0,
                "cold_peak_device_bytes": 100,
                "handoff_peak_device_bytes": 90,
                "repetitions": 1,
            },
        }

    return run


def _execute(
    tmp_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    source_blob: str,
    backend: Any,
) -> dict[str, Any]:
    return runtime.execute(
        config,
        producer_repository_root=tmp_path / "branch-advanced-or-detached",
        consumer_receipt_path=tmp_path / "consumer.json",
        consumer_receipt_sidecar_path=tmp_path / "consumer.json.sha256",
        consumer_receipt_schema_path=tmp_path / "consumer.schema.json",
        source_blob_path=source_blob,
        record_index=7,
        adapter_receipt_path=paths["receipt"],
        adapter_receipt_sidecar_path=paths["sidecar"],
        adapter_receipt_schema_path=paths["schema"],
        model_dir=tmp_path / "model",
        planner_adapter_name="planner_q_only",
        output_adapter_name="tool_q_only",
        gpu_lock_path=tmp_path / "gpu.lock",
        receipt_path=tmp_path / "receipt.json",
        source_gate=_source_gate(config, source_blob),
        backend=backend,
    )


def test_checked_config_is_strict_model_free_and_claims_are_narrow() -> None:
    config = runtime.load_config()
    receipt = runtime.build_status(config)
    assert receipt["status"] == "blocked_model_free"
    assert receipt["resource_counters"]["gpu_requests"] == 0
    assert receipt["resource_counters"]["model_loads"] == 0
    assert config["source_gate"]["current_head_or_upstream_required"] is False
    assert config["adapter_receipt"]["schema_version"] == (
        "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-runtime-run-receipt.v2"
    )
    assert config["adapter_receipt"]["required_mode"] == "full"
    assert tuple(config["adapter_receipt"]["required_arms"]) == runtime.TRAINED_ARMS
    assert all(value == "pending" for value in config["teacher_final"].values())
    assert config["execution"]["atomic_create_once"] is True
    assert config["claims"]["exact_prefix_value_handoff_requires_physical_pass"] is True
    assert config["claims"]["prefill_compute_handoff_requires_physical_pass"] is True
    assert "exact_prefix_value_handoff" not in config["claims"]
    assert "prefill_compute_handoff" not in config["claims"]
    assert {
        item for item in receipt["failure_codes"] if item.startswith("teacher_final.")
    } == {f"teacher_final.{name}" for name in runtime.TEACHER_IDENTITY_FIELDS}
    assert config["tokenizer_binding"]["runtime_special_token_overlay"] == {
        "canonical_files_modified": False,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "start_of_turn_token_id": 105,
        "end_of_turn_token_id": 106,
        "start_and_end_marked_special_in_memory": True,
    }
    for name in (
        "shared_storage",
        "zero_copy",
        "rdma",
        "full_generation_sharing",
        "pointer_identity_is_proof",
        "formal",
        "live",
    ):
        assert receipt["claims"][name] is False


def test_config_and_receipt_schemas_are_draft_2020_12() -> None:
    for path in (runtime.CONFIG_SCHEMA_PATH, runtime.RECEIPT_SCHEMA_PATH):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


def test_topology_is_derived_from_actual_layer_types() -> None:
    config = runtime.load_config()
    topology = runtime.derive_layer_topology(_model_config(), config["model"])
    assert topology["layers"] == 26
    assert topology["sliding_layers"] == 22
    assert topology["full_layers"] == 4
    assert topology["full_layer_indices"] == [5, 11, 17, 23]
    mutated = _model_config()
    mutated["layer_types"][0] = "full_attention"
    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_layer_topology_invalid",
    ):
        runtime.derive_layer_topology(mutated, config["model"])


def test_retained_prefix_layout_and_slice_evidence_cover_both_boundaries() -> None:
    import torch

    sliding = {
        "index": 0,
        "attention_kind": "sliding",
        "window": 512,
    }
    full = {
        "index": 0,
        "attention_kind": "full",
        "window": None,
    }
    assert runtime._retained_prefix_layout(
        sliding,
        source_tokens=511,
        tail_tokens=1,
    ) == (511, 510)
    assert runtime._retained_prefix_layout(
        sliding,
        source_tokens=511,
        tail_tokens=2,
    ) == (511, 509)
    assert runtime._retained_prefix_layout(
        full,
        source_tokens=600,
        tail_tokens=2,
    ) == (602, 600)

    source_key = torch.arange(1022, dtype=torch.float32).reshape(1, 1, 511, 2)
    source_value = source_key + 2048
    planner_key = torch.cat(
        (source_key[..., -510:, :], torch.zeros(1, 1, 1, 2)),
        dim=-2,
    )
    planner_value = torch.cat(
        (source_value[..., -510:, :], torch.ones(1, 1, 1, 2)),
        dim=-2,
    )
    output_key = torch.cat(
        (source_key[..., -509:, :], torch.zeros(1, 1, 2, 2)),
        dim=-2,
    )
    output_value = torch.cat(
        (source_value[..., -509:, :], torch.ones(1, 1, 2, 2)),
        dim=-2,
    )
    source_cache = SimpleNamespace(
        key_cache=[source_key],
        value_cache=[source_value],
    )
    planner_cache = SimpleNamespace(
        key_cache=[planner_key],
        value_cache=[planner_value],
    )
    output_cache = SimpleNamespace(
        key_cache=[output_key],
        value_cache=[output_value],
    )
    topology = {
        "layers": 1,
        "layers_detail": [sliding],
    }
    planner = runtime._branch_prefix_evidence(
        source_cache,
        planner_cache,
        topology,
        stage="planner",
        tail_tokens=1,
        torch_module=torch,
    )
    output = runtime._branch_prefix_evidence(
        source_cache,
        output_cache,
        topology,
        stage="output",
        tail_tokens=2,
        torch_module=torch,
    )
    assert planner[0]["planner_prefix_value_equal"] is True
    assert planner[0]["planner_retained_prefix_tokens"] == 510
    assert output[0]["output_prefix_value_equal"] is True
    assert output[0]["output_retained_prefix_tokens"] == 509


def test_boundary_rejects_position_mask_and_adapter_drift() -> None:
    base = {name: _sha(name) for name in runtime.LINEAGE_FIELDS}
    base["producer_candidate_commit"] = "1" * 40
    base["producer_release_commit"] = "2" * 40
    base["producer_release_tree"] = "3" * 40
    commit = runtime.boundary_commit_sha256(base)
    runtime.validate_boundary(base, dict(base), commit)
    for field, code in (
        ("position_sha256", "kv_v2_wrong_position"),
        ("attention_mask_sha256", "kv_v2_wrong_attention_mask"),
        ("adapter_lineage_sha256", "kv_v2_adapter_switch_invalid"),
    ):
        mutated = dict(base)
        mutated[field] = _sha(f"mutated-{field}")
        with pytest.raises(runtime.SharedPrefixKVRuntimeError, match=code):
            runtime.validate_boundary(base, mutated, commit)


def test_teacher_final_must_match_the_unique_configured_release() -> None:
    expected = {
        "schema_version": "unit.teacher-final.v2",
        "binding_sha256": _sha("teacher-binding"),
        "manifest_sha256": _sha("teacher-manifest"),
        "record_schema_sha256": _sha("teacher-record-schema"),
        "release_receipt_sha256": _sha("teacher-release-receipt"),
        "release_attestation_sha256": _sha("teacher-release-attestation"),
        "shard_inventory_sha256": _sha("teacher-shard-inventory"),
        "public_identity_sha256": _sha("teacher-public-identity"),
    }
    teacher = {
        **expected,
        **runtime.TEACHER_FINAL_METRICS,
        "producer_original_target_fallback": False,
    }
    runtime._validate_teacher_final_identity(teacher, expected)

    for stale_version in (
        "anchor.gemma3-chat-unbalanced-v2-teacher-final-binding.v1",
        "anchor.gemma3-chat-unbalanced-v2-teacher-final-binding.v2",
        "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding.v2",
    ):
        stale = dict(teacher)
        stale["schema_version"] = stale_version
        with pytest.raises(
            runtime.SharedPrefixKVRuntimeError,
            match="kv_v2_adapter_receipt_invalid",
        ):
            runtime._validate_teacher_final_identity(stale, expected)


def test_mock_execute_passes_without_current_head_or_upstream_dependency(
    tmp_path: Path,
) -> None:
    config, paths, _ = _ready_config(tmp_path)
    source_blob = "components/router/records.jsonl"
    identity = _source_identity(config, source_blob)
    receipt = _execute(
        tmp_path,
        config,
        paths,
        source_blob,
        _backend(config, identity),
    )
    assert receipt["status"] == "passed_diagnostic_prefix_handoff"
    assert receipt["source"]["producer_release_tree"] == (
        "9bf154d481623b696dbc8a4bdf276967710b7214"
    )
    assert "current_head" not in receipt["source"]
    assert "upstream_commit" not in receipt["source"]
    assert receipt["comparison"]["next_token_argmax_equal"] is True
    assert receipt["isolation"]["source_cache_unchanged"] is True
    assert receipt["claims"]["shared_storage"] is False
    assert receipt["claims"]["zero_copy"] is False
    assert receipt["claims"]["rdma"] is False
    raw = (tmp_path / "receipt.json").read_text(encoding="utf-8")
    assert "serialized_prefix_text" not in raw
    assert '"raw_logits":' not in raw
    assert '"raw_token_ids":' not in raw
    assert "input_ids" not in raw


def test_source_commit_or_blob_drift_is_rejected_before_backend(
    tmp_path: Path,
) -> None:
    config, _, _ = _ready_config(tmp_path)
    source_blob = "components/tool_eval/records.jsonl"
    identity = _source_identity(config, source_blob)
    identity["producer_release_commit"] = "f" * 40

    def bad_gate(**_: Any) -> dict[str, Any]:
        return {
            "prefix_input_ids": (2, 105, 2364, 107, 106, 1, 107, 105, 4368, 107),
            "source_identity": identity,
        }

    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_source_identity_drift",
    ):
        runtime.authenticate_source(
            config,
            producer_repository_root=tmp_path,
            consumer_receipt_path=tmp_path / "consumer.json",
            consumer_receipt_sidecar_path=tmp_path / "consumer.json.sha256",
            consumer_receipt_schema_path=tmp_path / "consumer.schema.json",
            model_root=tmp_path / "model",
            source_blob_path=source_blob,
            record_index=7,
            source_gate=bad_gate,
        )


def test_boundary_drift_preserves_body_free_failure_receipt(
    tmp_path: Path,
) -> None:
    config, paths, _ = _ready_config(tmp_path)
    source_blob = "components/planner_eval/records.jsonl"
    identity = _source_identity(config, source_blob)
    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_boundary_digest_mismatch",
    ):
        _execute(
            tmp_path,
            config,
            paths,
            source_blob,
            _backend(config, identity, boundary_drift=True),
        )
    failure = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed_diagnostic_prefix_handoff"
    assert failure["failure_codes"] == ["kv_v2_boundary_digest_mismatch"]
    assert failure["claims"]["exact_prefix_value_handoff"] is False
    assert failure["resource_counters"]["raw_logits_persisted"] == 0
    assert failure["resource_counters"]["raw_token_ids_persisted"] == 0


def test_private_tail_alias_fails_closed(tmp_path: Path) -> None:
    config, paths, _ = _ready_config(tmp_path)
    source_blob = "eval_proxy/chat.jsonl"
    identity = _source_identity(config, source_blob)
    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_private_tail_alias",
    ):
        _execute(
            tmp_path,
            config,
            paths,
            source_blob,
            _backend(config, identity, private_alias=True),
        )


@pytest.mark.parametrize(
    ("backend_kwargs", "failure_code"),
    [
        ({"source_mutated": True}, "kv_v2_cache_mutation"),
        ({"prefix_mismatch": True}, "kv_v2_prefix_value_mismatch"),
        ({"output_prefix_mismatch": True}, "kv_v2_prefix_value_mismatch"),
    ],
)
def test_cache_or_prefix_value_drift_fails_closed(
    tmp_path: Path,
    backend_kwargs: dict[str, bool],
    failure_code: str,
) -> None:
    config, paths, _ = _ready_config(tmp_path)
    source_blob = "components/router/records.jsonl"
    identity = _source_identity(config, source_blob)
    with pytest.raises(runtime.SharedPrefixKVRuntimeError, match=failure_code):
        _execute(
            tmp_path,
            config,
            paths,
            source_blob,
            _backend(config, identity, **backend_kwargs),
        )
    failure = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert failure["failure_codes"] == [failure_code]


def test_adapter_file_drift_is_rejected_before_backend(tmp_path: Path) -> None:
    config, paths, _ = _ready_config(tmp_path)
    adapter_model = tmp_path / "adapters" / "tool_q_only" / "adapter_model.safetensors"
    adapter_model.write_bytes(b"tampered-adapter\n")
    called = {"backend": 0}

    def backend(_: Mapping[str, Any]) -> dict[str, Any]:
        called["backend"] += 1
        return {}

    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_adapter_receipt_invalid",
    ):
        _execute(
            tmp_path,
            config,
            paths,
            "components/tool_eval/records.jsonl",
            backend,
        )
    failure = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert called["backend"] == 0
    assert failure["failure_codes"] == ["kv_v2_adapter_receipt_invalid"]
    assert failure["resource_counters"]["model_loads"] == 0
    assert failure["resource_counters"]["gpu_requests"] == 0


def test_default_execute_is_blocked_before_source_or_backend() -> None:
    config = runtime.load_config()
    called = {"source": 0, "backend": 0}

    def source(**_: Any) -> dict[str, Any]:
        called["source"] += 1
        return {}

    def backend(_: Mapping[str, Any]) -> dict[str, Any]:
        called["backend"] += 1
        return {}

    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_execute_not_authorized",
    ):
        runtime.execute(
            config,
            producer_repository_root="unused",
            consumer_receipt_path="unused",
            consumer_receipt_sidecar_path="unused",
            consumer_receipt_schema_path="unused",
            source_blob_path="components/router/records.jsonl",
            record_index=0,
            adapter_receipt_path="unused",
            adapter_receipt_sidecar_path="unused",
            adapter_receipt_schema_path="unused",
            model_dir="unused",
            planner_adapter_name="planner_q_only",
            output_adapter_name="tool_q_only",
            gpu_lock_path="unused",
            receipt_path="unused",
            source_gate=source,
            backend=backend,
        )
    assert called == {"source": 0, "backend": 0}


def test_receipt_publish_is_atomic_create_once(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    original = b'{"owner":"existing"}\n'
    target.write_bytes(original)
    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_atomic_publish_failed",
    ):
        runtime._atomic_write(target, {"owner": "replacement"})
    assert target.read_bytes() == original
    assert not tuple(tmp_path.glob(".receipt.json.*.tmp"))


def test_gpu_lock_existing_target_and_replacement_are_not_deleted(
    tmp_path: Path,
) -> None:
    target = tmp_path / "gpu.lock"
    target.write_bytes(b"existing\n")
    with pytest.raises(
        runtime.SharedPrefixKVRuntimeError,
        match="kv_v2_gpu_lock_busy",
    ):
        with runtime._gpu_lock(target):
            pytest.fail("existing lock must not be acquired")
    assert target.read_bytes() == b"existing\n"

    owned = runtime._owned_path_identity(
        target,
        require_directory=False,
        code="kv_v2_gpu_lock_invalid",
    )
    target.unlink()
    target.write_bytes(b"replacement\n")
    assert runtime._release_owned_path(target, owned) is False
    assert target.read_bytes() == b"replacement\n"
