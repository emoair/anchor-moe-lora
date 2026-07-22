"""Metadata-only, fail-closed consumer for the Qwen toy prerequisite.

The companion consumer authenticates producer contracts and hash-only source-ID
inventories.  It deliberately never opens the toy diagnostic records or any
protected Gold, held-out, SWE-bench, or scaffold body file.  A verified result
therefore describes diagnostic metadata only and can never authorize training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import yaml


CONFIG_VERSION = "anchor.qwen-toy-prerequisite-consumer-config.v1"
CONFIG_SHA256 = "ac2c522015798b379566c8c2aa96e5689398fb303240db916818c3a06e667811"
PRODUCER_COMMIT = "744e23f975b13923903f5fabe04c32e74ea25dc4"
PRODUCER_ID = "anchor.qwen-toy-prerequisite-producer.v1"
MAIN_VERSION = "anchor.qwen-toy-prerequisite-manifest.v1"
INVENTORY_VERSION = "anchor.protected-source-id-inventory.v1"
TRIGGER_VERSION = "anchor.qwen-request-local-trigger-materialization.v1"

INVENTORY_SCHEMA_SHA256 = (
    "1f536b788bd4255b633f7f162f3d731ab0bd2d0c7e67fcf0c31892c2e6155d49"
)
RECORD_SCHEMA_SHA256 = (
    "1d63f1ce8134060b60a73ae0aac3b3574d817776d678723f8aa5717cb69d834e"
)
MAIN_SCHEMA_SHA256 = "b55a0200a3945189687dc0363915e5911bbef41eb6aedcf0cb0f0ceb5bb18e20"
TRIGGER_SCHEMA_SHA256 = (
    "8a8d97c1ef1513999e215fa63883d476ad7d062e7bcff8274971b2388e9c62e9"
)
PRODUCER_CONFIG_SHA256 = (
    "68bbaa13068ea591ab6f26bc31f4077967c05b6a86249811538049f19c798bd8"
)
CLOSED_GRAMMAR_SHA256 = (
    "514ddeb93b6f2afbcd99dbfd81d5fedaed93627957a138b6f07a27cb3c1deab9"
)
GENERATOR_IMPLEMENTATION_SHA256 = (
    "b6c3c510e552c265c833d47950b9201da4a97998ca6165e99b972ba6d9978c55"
)
ATTESTER_IMPLEMENTATION_SHA256 = (
    "01009bf18b164282c9006e4c2a42432a51d4566da414ead7454f783832c60ec3"
)
BUILDER_IMPLEMENTATION_SHA256 = (
    "c76e72114d2cea8f5a6e1941275002565ae24b01c9e3e9976dd8a6829e27a6e7"
)
MAIN_MANIFEST_SHA256 = (
    "99b94d71639e252c2d768b84a444efa09e844d287c691d8ddfa8312481f2f311"
)
MAIN_SIDECAR_SHA256 = "b8a3f7f7bec390da842ef35f8c9942a985051400c8e65857d6ba1a906b23c951"
AUDIT_SHA256 = "1c8e3dc84d99c3bb92019ca1a743f3429fa078a9ba150b39564027d43e07d18c"
AUDIT_SIDECAR_SHA256 = (
    "055fa0251f2cde420846203c0a5d078e4ff1a3c001f06173d9fa3841c032925b"
)
TOY_RECORDS_DECLARED_SHA256 = (
    "7253826d721e8c91c35926aacac751266589d04056c49f252bde0b613b2b4507"
)
TOY_SOURCE_IDS_FILE_SHA256 = (
    "2135cea977ac9b0e9685e779e0dc3ad78f51445452db55e0875dc485e0f5b568"
)
TOY_SOURCE_ID_INVENTORY_SHA256 = (
    "38774bdd155ca070e3af41c34eb297ff0d185ee009d21cccbe52b928f4880e57"
)
PROTECTED_INVENTORY_SET_SHA256 = (
    "d0bd5702a9c6bbbb1db547b826a94d960a518e1f6ef3e60bfdd25dcd93a3fe22"
)
GENERATION_READ_SET_SHA256 = (
    "1cc3080e8a7f84770042b173b9efb8166372ec346768ae067b2f2dc2dbf1ce26"
)

ORDERED_SOURCE_CLASSES = (
    "swebench_source",
    "gold_partition",
    "partial_gold_export",
    "heldout",
    "legacy_heldout_cases",
    "synthetic_scaffold",
)
READY_SOURCE_CLASSES = ("swebench_source", "heldout")
UNAVAILABLE_SOURCE_CLASSES = (
    "gold_partition",
    "partial_gold_export",
    "legacy_heldout_cases",
    "synthetic_scaffold",
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _InventoryBinding:
    manifest_sha256: str
    sidecar_sha256: str
    status: str
    source_id_count: int | None = None
    source_ids_file_sha256: str | None = None
    source_id_inventory_sha256: str | None = None


INVENTORY_BINDINGS: Mapping[str, _InventoryBinding] = {
    "swebench_source": _InventoryBinding(
        manifest_sha256="52bdf58cb903c807ba66a0c427ec5a968854164f4be8f54e3c79f3ff7b81d365",
        sidecar_sha256="ae62ef11ab4db0546f2e5910030731778d2ca1c09cb6769b39d83b711076ff32",
        status="ready",
        source_id_count=19008,
        source_ids_file_sha256="9d068d921795f7ffcafbb88c9b029e47d1d3253934c163998ef01ad72d378a95",
        source_id_inventory_sha256=(
            "7cbc7039203e9e4cec2b69182968436c637a1cf803d51106c44df529306ccea4"
        ),
    ),
    "gold_partition": _InventoryBinding(
        manifest_sha256="b5a79f3ace4b6bd75fa942c03bb255c0617d7da4083c5e55a4b43aeeb8d539da",
        sidecar_sha256="48a2cab39650d02c9fa06eacf47663b00c0bc5ee8bf565fb4ae5131ef8b96d93",
        status="unavailable",
    ),
    "partial_gold_export": _InventoryBinding(
        manifest_sha256="373a96f666ae7ce00d1a23fe4b1488c39d5ab191cdd2c2099c1466bc2454e9c3",
        sidecar_sha256="e47c29b8f844f6a7f1e472611f5043288540a3c38c80ea8e171148f975536f63",
        status="unavailable",
    ),
    "heldout": _InventoryBinding(
        manifest_sha256="f2d9f09539620e3a41abf3d610d4bedc74c1a32a902590840fdc8f207873534f",
        sidecar_sha256="9a55bdc070782fc8ab751c400217432a0454d6311296a361a2d7918c7536192c",
        status="ready",
        source_id_count=6,
        source_ids_file_sha256="19262eaf2e860b4c8dfdfc1c1277869e3e58f68f087441d43bbccffe04ecd833",
        source_id_inventory_sha256=(
            "dabdee98a6e7624e0befa025b19a77d0c39830613e96088a20dd69f1c201c267"
        ),
    ),
    "legacy_heldout_cases": _InventoryBinding(
        manifest_sha256="cf94b6da7cc05cdad1c8dcb2c935218aebc4b3a556c7916bff07a12d3450df84",
        sidecar_sha256="ffe26c7d49df084ebdc86f605a266491b58b5dc54e8d4f89bf255394ea56748d",
        status="unavailable",
    ),
    "synthetic_scaffold": _InventoryBinding(
        manifest_sha256="c1cfadd240c989fcea3041a6b49b0a8aae674d72115aaa74a076553544207456",
        sidecar_sha256="f95c733dad2ebc6b5d0b1da2789d32530d4e35955b1813d21d28679c4fe657da",
        status="unavailable",
    ),
}

READ_WHITELIST = (
    "configs/research/qwen_toy_prerequisite_consumer_v1.yaml",
    "configs/research/protected_source_id_inventory.schema.json",
    "configs/research/qwen_toy_diagnostic_record.schema.json",
    "configs/research/qwen_toy_prerequisite_manifest.schema.json",
    "configs/research/qwen_request_local_trigger_materialization.schema.json",
    "configs/research/qwen_toy_prerequisite_v1.json",
    "configs/research/qwen_toy_closed_grammar_v1.json",
    "fixtures/research/qwen_toy_prerequisite_v1/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/manifest.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/audit.json",
    "fixtures/research/qwen_toy_prerequisite_v1/audit.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/toy/source_ids.sha256.jsonl",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/swebench_source/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/swebench_source/manifest.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/swebench_source/source_ids.sha256.jsonl",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/gold_partition/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/gold_partition/manifest.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/partial_gold_export/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/partial_gold_export/manifest.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/heldout/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/heldout/manifest.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/heldout/source_ids.sha256.jsonl",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/legacy_heldout_cases/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/legacy_heldout_cases/manifest.json.sha256",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/synthetic_scaffold/manifest.json",
    "fixtures/research/qwen_toy_prerequisite_v1/inventories/synthetic_scaffold/manifest.json.sha256",
)


class QwenToyPrerequisiteConsumerError(RuntimeError):
    """Raised when metadata cannot be authenticated without protected reads."""


def _fail(code: str) -> None:
    raise QwenToyPrerequisiteConsumerError(code)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@dataclass(frozen=True)
class _BytesSnapshot:
    relative_path: str
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    return stat.S_ISLNK(value.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _assert_physical_ancestry(path: Path, root: Path, code: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(code)
    current = root
    if _is_reparse_or_symlink(current):
        _fail(code)
    for part in relative.parts:
        current /= part
        if current.exists() and _is_reparse_or_symlink(current):
            _fail(code)


def _read_bytes_snapshot(root: Path, relative_path: str, code: str) -> _BytesSnapshot:
    relative = Path(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        _fail(code)
    lexical = root / relative
    _assert_physical_ancestry(lexical, root, code)
    path = lexical.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError:
        _fail(code)
    try:
        if not path.is_file() or path.is_symlink():
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise QwenToyPrerequisiteConsumerError(code) from exc
    before_id = _stat_identity(before)
    after_id = _stat_identity(after)
    if (
        before_id != after_id
        or after_id != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(
        relative_path=relative.as_posix(),
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        identity=after_id,
    )


class _ReadLedger:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.allowed = frozenset(READ_WHITELIST)
        self.snapshots: dict[str, _BytesSnapshot] = {}

    def read(self, relative_path: str, code: str) -> _BytesSnapshot:
        normalized = Path(relative_path.replace("\\", "/")).as_posix()
        if normalized not in self.allowed:
            _fail("read_path_not_whitelisted")
        if normalized in self.snapshots:
            _fail("duplicate_initial_snapshot_read")
        snapshot = _read_bytes_snapshot(self.root, normalized, code)
        self.snapshots[normalized] = snapshot
        return snapshot

    def final_recheck(self) -> None:
        for relative, initial in self.snapshots.items():
            current = _read_bytes_snapshot(
                self.root, relative, "artifact_changed_during_validation"
            )
            if (
                current.identity != initial.identity
                or current.sha256 != initial.sha256
                or current.data != initial.data
            ):
                _fail("artifact_changed_during_validation")
            if relative.endswith((".json", ".schema.json")):
                if _decode_json(current, "final_json_reparse_failed") != _decode_json(
                    initial, "initial_json_reparse_failed"
                ):
                    _fail("artifact_reparse_drift")
            elif relative.endswith((".yaml", ".yml")):
                if _decode_yaml(current, "final_yaml_reparse_failed") != _decode_yaml(
                    initial, "initial_yaml_reparse_failed"
                ):
                    _fail("artifact_reparse_drift")
            elif relative.endswith("source_ids.sha256.jsonl"):
                if _decode_hash_id_lines(
                    current, "final_hash_inventory_reparse_failed"
                ) != _decode_hash_id_lines(
                    initial, "initial_hash_inventory_reparse_failed"
                ):
                    _fail("artifact_reparse_drift")


def _decode_json(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        first = json.loads(snapshot.data.decode("utf-8"))
        second = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenToyPrerequisiteConsumerError(code) from exc
    if first != second:
        _fail(f"{code}_same_bytes_reparse_drift")
    return _mapping(first, code)


def _decode_yaml(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        first = yaml.safe_load(snapshot.data.decode("utf-8"))
        second = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise QwenToyPrerequisiteConsumerError(code) from exc
    if first != second:
        _fail(f"{code}_same_bytes_reparse_drift")
    return _mapping(first, code)


def _decode_hash_id_lines(snapshot: _BytesSnapshot, code: str) -> tuple[str, ...]:
    try:
        text = snapshot.data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise QwenToyPrerequisiteConsumerError(code) from exc
    if not text.endswith("\n") or "\r" in text:
        _fail(f"{code}_noncanonical_line_endings")
    lines = tuple(text[:-1].split("\n"))
    if not lines or any(_HEX64.fullmatch(line) is None for line in lines):
        _fail(f"{code}_invalid_identifier")
    if tuple(sorted(set(lines))) != lines:
        _fail(f"{code}_not_sorted_unique")
    return lines


def _logical_inventory_sha256(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _validate_schema_document(
    snapshot: _BytesSnapshot, expected: str, code: str
) -> None:
    if snapshot.sha256 != expected:
        _fail(f"{code}_sha256_mismatch")
    schema = _decode_json(snapshot, f"{code}_json_invalid")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except ImportError as exc:
        raise QwenToyPrerequisiteConsumerError(
            f"{code}_jsonschema_dependency_unavailable"
        ) from exc
    except Exception as exc:
        raise QwenToyPrerequisiteConsumerError(f"{code}_invalid") from exc


def _validate_instance(
    schema_snapshot: _BytesSnapshot,
    expected_schema_sha256: str,
    instance: Mapping[str, Any],
    code: str,
) -> None:
    if schema_snapshot.sha256 != expected_schema_sha256:
        _fail(f"{code}_schema_sha256_mismatch")
    schema = _decode_json(schema_snapshot, f"{code}_schema_invalid")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator(schema).validate(instance)
    except ImportError as exc:
        raise QwenToyPrerequisiteConsumerError(
            f"{code}_jsonschema_dependency_unavailable"
        ) from exc
    except Exception as exc:
        raise QwenToyPrerequisiteConsumerError(
            f"{code}_schema_validation_failed"
        ) from exc


def _validate_sidecar(
    sidecar: _BytesSnapshot,
    *,
    target_sha256: str,
    target_name: str,
    physical_sha256: str,
    code: str,
) -> None:
    if sidecar.sha256 != physical_sha256:
        _fail(f"{code}_physical_sha256_mismatch")
    expected = f"{target_sha256}  {target_name}\n".encode("ascii")
    if sidecar.data != expected:
        _fail(f"{code}_noncanonical")


def _expected_bindings() -> dict[str, object]:
    return {
        "protected_inventory_schema_sha256": INVENTORY_SCHEMA_SHA256,
        "toy_record_schema_sha256": RECORD_SCHEMA_SHA256,
        "main_manifest_schema_sha256": MAIN_SCHEMA_SHA256,
        "trigger_receipt_schema_sha256": TRIGGER_SCHEMA_SHA256,
        "producer_config_sha256": PRODUCER_CONFIG_SHA256,
        "closed_grammar_sha256": CLOSED_GRAMMAR_SHA256,
        "generator_implementation_sha256": GENERATOR_IMPLEMENTATION_SHA256,
        "attester_implementation_sha256": ATTESTER_IMPLEMENTATION_SHA256,
        "builder_implementation_sha256": BUILDER_IMPLEMENTATION_SHA256,
        "main_manifest_sha256": MAIN_MANIFEST_SHA256,
        "main_sidecar_physical_sha256": MAIN_SIDECAR_SHA256,
        "audit_sha256": AUDIT_SHA256,
        "audit_sidecar_physical_sha256": AUDIT_SIDECAR_SHA256,
        "toy_records_declared_sha256": TOY_RECORDS_DECLARED_SHA256,
        "toy_source_ids_file_sha256": TOY_SOURCE_IDS_FILE_SHA256,
        "toy_source_id_inventory_sha256": TOY_SOURCE_ID_INVENTORY_SHA256,
        "protected_inventory_set_sha256": PROTECTED_INVENTORY_SET_SHA256,
        "generation_read_set_sha256": GENERATION_READ_SET_SHA256,
    }


def _expected_paths() -> dict[str, str]:
    return {
        "protected_inventory_schema": (
            "configs/research/protected_source_id_inventory.schema.json"
        ),
        "toy_record_schema": "configs/research/qwen_toy_diagnostic_record.schema.json",
        "main_manifest_schema": (
            "configs/research/qwen_toy_prerequisite_manifest.schema.json"
        ),
        "trigger_receipt_schema": (
            "configs/research/qwen_request_local_trigger_materialization.schema.json"
        ),
        "producer_config": "configs/research/qwen_toy_prerequisite_v1.json",
        "closed_grammar": "configs/research/qwen_toy_closed_grammar_v1.json",
        "artifact_root": "fixtures/research/qwen_toy_prerequisite_v1",
        "main_manifest": "fixtures/research/qwen_toy_prerequisite_v1/manifest.json",
        "audit": "fixtures/research/qwen_toy_prerequisite_v1/audit.json",
    }


def _validate_consumer_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "claim_scope",
        "producer",
        "bindings",
        "paths",
        "inventory_contract",
        "read_whitelist",
        "policy",
    }:
        _fail("consumer_config_shape_invalid")
    if config.get("schema_version") != CONFIG_VERSION:
        _fail("consumer_config_version_mismatch")
    if config.get("claim_scope") != (
        "metadata_only_diagnostic_prerequisite_consumer_no_training_authority"
    ):
        _fail("consumer_claim_scope_drift")
    if dict(_mapping(config.get("producer"), "consumer_producer_invalid")) != {
        "commit": PRODUCER_COMMIT,
        "producer_id": PRODUCER_ID,
        "main_manifest_version": MAIN_VERSION,
        "inventory_version": INVENTORY_VERSION,
        "trigger_receipt_version": TRIGGER_VERSION,
    }:
        _fail("consumer_producer_identity_drift")
    if dict(_mapping(config.get("bindings"), "consumer_bindings_invalid")) != (
        _expected_bindings()
    ):
        _fail("consumer_binding_identity_drift")
    if (
        dict(_mapping(config.get("paths"), "consumer_paths_invalid"))
        != _expected_paths()
    ):
        _fail("consumer_paths_drift")
    inventory = _mapping(
        config.get("inventory_contract"), "consumer_inventory_contract_invalid"
    )
    if dict(inventory) != {
        "ordered_source_classes": list(ORDERED_SOURCE_CLASSES),
        "ready_source_classes": list(READY_SOURCE_CLASSES),
        "unavailable_source_classes": list(UNAVAILABLE_SOURCE_CLASSES),
        "ready_count": 2,
        "total_count": 6,
    }:
        _fail("consumer_inventory_contract_drift")
    if tuple(config.get("read_whitelist", ())) != READ_WHITELIST:
        _fail("consumer_read_whitelist_drift")
    if dict(_mapping(config.get("policy"), "consumer_policy_invalid")) != {
        "metadata_only": True,
        "fail_closed": True,
        "toy_record_body_read": False,
        "protected_content_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "training_authorized": False,
        "formal_training_authorized": False,
        "zero_intersection_claimed": False,
        "v1_attestation_emitted": False,
    }:
        _fail("consumer_policy_drift")


def _validate_producer_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "anchor.qwen-toy-prerequisite-config.v1":
        _fail("producer_config_version_drift")
    if config.get("producer_id") != PRODUCER_ID:
        _fail("producer_config_identity_drift")
    generator = _mapping(config.get("generator"), "producer_generator_invalid")
    if generator.get("output_file") != "toy/diagnostic.jsonl":
        _fail("producer_diagnostic_path_drift")
    if generator.get("record_count") != 8 or generator.get("partition") != (
        "diagnostic_only"
    ):
        _fail("producer_generator_contract_drift")
    safety = _mapping(config.get("safety"), "producer_safety_invalid")
    if safety != {
        "diagnostic_only": True,
        "formal_training_authorized": False,
        "consumable_by_formal_release": False,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_content_reads": 0,
    }:
        _fail("producer_safety_drift")


def _validate_pending_trigger(
    trigger: Mapping[str, Any], trigger_schema: _BytesSnapshot
) -> None:
    _validate_instance(
        trigger_schema, TRIGGER_SCHEMA_SHA256, trigger, "trigger_receipt"
    )
    nullable = (
        "tokenizer_binding_sha256",
        "chat_template_sha256",
        "exact_r2_serialization_sha256",
        "ordered_input_token_ids_sha256",
        "trigger_span_zero_based_exclusive",
        "boundary_overhang",
    )
    if trigger.get("schema_version") != TRIGGER_VERSION:
        _fail("trigger_receipt_version_drift")
    if trigger.get("status") != "pending_request_local_materialization":
        _fail("trigger_receipt_must_remain_pending")
    if any(trigger.get(field) is not None for field in nullable):
        _fail("trigger_receipt_unbound_identity_must_be_null")
    false_fields = (
        "isolated_trigger_encoding_authoritative",
        "global_token_index_emitted",
        "token_ids_emitted",
        "planner_request1_private_kv_reused",
        "formal_training_authorized",
    )
    if any(trigger.get(field) is not False for field in false_fields):
        _fail("trigger_receipt_false_boundary_drift")
    if trigger.get("full_r2_single_tokenization_required") is not True:
        _fail("trigger_receipt_single_tokenization_requirement_drift")


def _validate_hash_inventory(
    snapshot: _BytesSnapshot,
    *,
    expected_file_sha256: str,
    expected_count: int,
    expected_logical_sha256: str,
    code: str,
) -> None:
    if snapshot.sha256 != expected_file_sha256:
        _fail(f"{code}_file_sha256_mismatch")
    lines = _decode_hash_id_lines(snapshot, code)
    if len(lines) != expected_count:
        _fail(f"{code}_count_mismatch")
    if _logical_inventory_sha256(lines) != expected_logical_sha256:
        _fail(f"{code}_logical_sha256_mismatch")


def _validate_main_semantics(
    main: Mapping[str, Any],
    *,
    producer_config: Mapping[str, Any],
    trigger_schema: _BytesSnapshot,
) -> None:
    if main.get("schema_version") != MAIN_VERSION:
        _fail("main_manifest_version_drift")
    if main.get("status") != ("toy_generation_verified_protected_inventory_incomplete"):
        _fail("main_manifest_status_drift")
    producer = _mapping(main.get("producer"), "main_producer_invalid")
    expected_producer_files = {
        "config": (
            "configs/research/qwen_toy_prerequisite_v1.json",
            PRODUCER_CONFIG_SHA256,
        ),
        "closed_grammar": (
            "configs/research/qwen_toy_closed_grammar_v1.json",
            CLOSED_GRAMMAR_SHA256,
        ),
        "inventory_schema": (
            "configs/research/protected_source_id_inventory.schema.json",
            INVENTORY_SCHEMA_SHA256,
        ),
        "record_schema": (
            "configs/research/qwen_toy_diagnostic_record.schema.json",
            RECORD_SCHEMA_SHA256,
        ),
        "manifest_schema": (
            "configs/research/qwen_toy_prerequisite_manifest.schema.json",
            MAIN_SCHEMA_SHA256,
        ),
        "trigger_receipt_schema": (
            "configs/research/qwen_request_local_trigger_materialization.schema.json",
            TRIGGER_SCHEMA_SHA256,
        ),
        "generator_implementation": (
            "src/anchor_mvp/swebench/toy_diagnostic_generator.py",
            GENERATOR_IMPLEMENTATION_SHA256,
        ),
        "attester_implementation": (
            "src/anchor_mvp/swebench/toy_diagnostic_auditor.py",
            ATTESTER_IMPLEMENTATION_SHA256,
        ),
        "builder_implementation": (
            "src/anchor_mvp/swebench/qwen_toy_prerequisite.py",
            BUILDER_IMPLEMENTATION_SHA256,
        ),
    }
    if producer.get("producer_id") != PRODUCER_ID:
        _fail("main_producer_identity_drift")
    for key, (path, sha256) in expected_producer_files.items():
        if _mapping(producer.get(key), f"main_{key}_invalid") != {
            "path": path,
            "sha256": sha256,
        }:
            _fail(f"main_{key}_identity_drift")

    toy = _mapping(main.get("toy"), "main_toy_invalid")
    if toy.get("record_count") != 8 or toy.get("partition") != "diagnostic_only":
        _fail("main_toy_count_or_partition_drift")
    if toy.get("tokenizer_bound") is not False:
        _fail("main_toy_must_remain_tokenizer_unbound")
    records = _mapping(toy.get("records"), "main_toy_records_invalid")
    if records.get("path") != "toy/diagnostic.jsonl" or records.get("sha256") != (
        TOY_RECORDS_DECLARED_SHA256
    ):
        _fail("main_toy_record_declaration_drift")
    source_ids = _mapping(toy.get("source_id_inventory"), "main_toy_source_ids_invalid")
    if source_ids != {
        "path": "toy/source_ids.sha256.jsonl",
        "sha256": TOY_SOURCE_IDS_FILE_SHA256,
        "bytes": 520,
        "records": 8,
    }:
        _fail("main_toy_source_ids_identity_drift")
    if toy.get("source_id_inventory_sha256") != TOY_SOURCE_ID_INVENTORY_SHA256:
        _fail("main_toy_logical_inventory_drift")

    read_set = _mapping(main.get("generation_read_set"), "main_read_set_invalid")
    if read_set.get("inventory_sha256") != GENERATION_READ_SET_SHA256:
        _fail("generation_read_set_identity_drift")
    if (
        read_set.get("protected_content_reads") != 0
        or read_set.get("unexpected_reads") != 0
    ):
        _fail("generation_read_set_safety_drift")
    expected_read_inputs = [
        (
            "generator_implementation",
            "src/anchor_mvp/swebench/toy_diagnostic_generator.py",
            GENERATOR_IMPLEMENTATION_SHA256,
        ),
        (
            "generator_config",
            "configs/research/qwen_toy_prerequisite_v1.json",
            PRODUCER_CONFIG_SHA256,
        ),
        (
            "closed_grammar",
            "configs/research/qwen_toy_closed_grammar_v1.json",
            CLOSED_GRAMMAR_SHA256,
        ),
    ]
    observed = [
        (item.get("role"), item.get("path"), item.get("sha256"))
        for item in read_set.get("inputs", ())
        if isinstance(item, Mapping)
    ]
    if observed != expected_read_inputs:
        _fail("generation_read_set_cross_binding_drift")

    _validate_pending_trigger(
        _mapping(main.get("request_local_trigger_binding"), "main_trigger_invalid"),
        trigger_schema,
    )
    proof = _mapping(main.get("proof"), "main_proof_invalid")
    if proof != {
        "coverage_ready_count": 2,
        "coverage_total": 6,
        "formal_training_authorized": False,
        "missing_source_classes": list(UNAVAILABLE_SOURCE_CLASSES),
        "protected_inventory_set_sha256": PROTECTED_INVENTORY_SET_SHA256,
        "status": "toy_generation_verified_protected_inventory_incomplete",
        "v1_attestation_emitted": False,
        "zero_intersection_claimed": False,
    }:
        _fail("main_proof_drift")
    execution = _mapping(main.get("execution"), "main_execution_invalid")
    if any(execution.get(field) != 0 for field in execution):
        _fail("main_execution_counter_drift")
    safety = _mapping(main.get("safety"), "main_safety_invalid")
    if safety != {
        "canonical_gold_written": False,
        "consumable_by_formal_release": False,
        "diagnostic_only": True,
        "formal_training_authorized": False,
        "heldout_written": False,
        "sample_content_emitted": False,
    }:
        _fail("main_safety_drift")

    configured_sources = producer_config.get("protected_sources", ())
    if (
        tuple(
            item.get("source_class")
            for item in configured_sources
            if isinstance(item, Mapping)
        )
        != ORDERED_SOURCE_CLASSES
    ):
        _fail("producer_source_order_drift")


def _validate_audit(main: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    expected_fields = {
        "attester",
        "auditor_version",
        "builder",
        "claim_scope",
        "final_recheck_passed",
        "formal_training_authorized",
        "generation_read_set_sha256",
        "gpu_requests",
        "independent_rebuild_passed",
        "model_loads",
        "network_requests",
        "protected_content_reads",
        "protected_inventory_coverage",
        "protected_inventory_set_sha256",
        "provider_requests",
        "record_count",
        "records_sha256",
        "same_snapshot_reparse_passed",
        "schema_version",
        "source_id_count",
        "source_id_inventory_file_sha256",
        "source_id_inventory_sha256",
        "status",
        "v1_attestation_emitted",
        "zero_intersection_claimed",
    }
    if set(audit) != expected_fields:
        _fail("audit_shape_invalid")
    if audit.get("schema_version") != "anchor.qwen-toy-diagnostic-audit.v1":
        _fail("audit_version_drift")
    if audit.get("status") != "passed":
        _fail("audit_status_drift")
    if audit.get("attester") != {
        "path": "src/anchor_mvp/swebench/toy_diagnostic_auditor.py",
        "sha256": ATTESTER_IMPLEMENTATION_SHA256,
    } or audit.get("builder") != {
        "path": "src/anchor_mvp/swebench/qwen_toy_prerequisite.py",
        "sha256": BUILDER_IMPLEMENTATION_SHA256,
    }:
        _fail("audit_implementation_cross_binding_drift")
    if audit.get("record_count") != 8 or audit.get("source_id_count") != 8:
        _fail("audit_count_drift")
    if audit.get("records_sha256") != TOY_RECORDS_DECLARED_SHA256:
        _fail("audit_record_declaration_drift")
    if audit.get("source_id_inventory_file_sha256") != TOY_SOURCE_IDS_FILE_SHA256:
        _fail("audit_source_id_file_drift")
    if audit.get("source_id_inventory_sha256") != TOY_SOURCE_ID_INVENTORY_SHA256:
        _fail("audit_source_id_logical_drift")
    if audit.get("protected_inventory_set_sha256") != PROTECTED_INVENTORY_SET_SHA256:
        _fail("audit_protected_set_drift")
    if audit.get("generation_read_set_sha256") != GENERATION_READ_SET_SHA256:
        _fail("audit_read_set_drift")
    coverage = _mapping(
        audit.get("protected_inventory_coverage"), "audit_coverage_invalid"
    )
    if coverage != {
        "missing_source_classes": list(UNAVAILABLE_SOURCE_CLASSES),
        "ready": 2,
        "total": 6,
    }:
        _fail("audit_coverage_drift")
    true_fields = (
        "final_recheck_passed",
        "independent_rebuild_passed",
        "same_snapshot_reparse_passed",
    )
    if any(audit.get(field) is not True for field in true_fields):
        _fail("audit_verification_flag_drift")
    false_fields = (
        "formal_training_authorized",
        "v1_attestation_emitted",
        "zero_intersection_claimed",
    )
    if any(audit.get(field) is not False for field in false_fields):
        _fail("audit_negative_claim_drift")
    zero_fields = (
        "gpu_requests",
        "model_loads",
        "network_requests",
        "protected_content_reads",
        "provider_requests",
    )
    if any(audit.get(field) != 0 for field in zero_fields):
        _fail("audit_counter_drift")
    main_audit = _mapping(main.get("audit"), "main_audit_invalid")
    if main_audit != {
        "final_recheck_passed": True,
        "independent_rebuild_passed": True,
        "path": "audit.json",
        "same_snapshot_reparse_passed": True,
        "sha256": AUDIT_SHA256,
        "sidecar_path": "audit.json.sha256",
        "sidecar_sha256": AUDIT_SIDECAR_SHA256,
    }:
        _fail("main_audit_cross_binding_drift")


def _inventory_relative(source_class: str, name: str) -> str:
    return (
        f"fixtures/research/qwen_toy_prerequisite_v1/inventories/{source_class}/{name}"
    )


def _validate_inventory_entry(
    *,
    source_class: str,
    entry: Mapping[str, Any],
    nested: Mapping[str, Any],
    binding: _InventoryBinding,
    inventory_schema: _BytesSnapshot,
    ledger: _ReadLedger,
) -> int:
    manifest_binding = _mapping(
        entry.get("manifest"), "inventory_manifest_binding_invalid"
    )
    sidecar_binding = _mapping(
        entry.get("sidecar"), "inventory_sidecar_binding_invalid"
    )
    if (
        entry.get("source_class") != source_class
        or entry.get("status") != binding.status
    ):
        _fail("inventory_main_status_or_order_drift")
    if manifest_binding != {
        "path": f"inventories/{source_class}/manifest.json",
        "sha256": binding.manifest_sha256,
    }:
        _fail(f"inventory_{source_class}_main_manifest_binding_drift")
    if sidecar_binding != {
        "path": f"inventories/{source_class}/manifest.json.sha256",
        "sha256": binding.sidecar_sha256,
    }:
        _fail(f"inventory_{source_class}_main_sidecar_binding_drift")
    if entry.get("sidecar_declared_sha256") != binding.manifest_sha256:
        _fail(f"inventory_{source_class}_declared_sidecar_drift")

    _validate_instance(
        inventory_schema, INVENTORY_SCHEMA_SHA256, nested, f"inventory_{source_class}"
    )
    if nested.get("schema_version") != INVENTORY_VERSION:
        _fail(f"inventory_{source_class}_version_drift")
    if nested.get("source_class") != source_class or nested.get("status") != (
        binding.status
    ):
        _fail(f"inventory_{source_class}_nested_status_drift")
    producer = _mapping(nested.get("producer"), "nested_inventory_producer_invalid")
    if producer.get("producer_id") != PRODUCER_ID:
        _fail(f"inventory_{source_class}_producer_id_drift")
    if producer.get("config") != {
        "path": "configs/research/qwen_toy_prerequisite_v1.json",
        "sha256": PRODUCER_CONFIG_SHA256,
    } or producer.get("implementation") != {
        "path": "src/anchor_mvp/swebench/qwen_toy_prerequisite.py",
        "sha256": BUILDER_IMPLEMENTATION_SHA256,
    }:
        _fail(f"inventory_{source_class}_producer_cross_binding_drift")
    if nested.get("inventory_schema") != {
        "path": "configs/research/protected_source_id_inventory.schema.json",
        "sha256": INVENTORY_SCHEMA_SHA256,
    }:
        _fail(f"inventory_{source_class}_schema_cross_binding_drift")
    safety = _mapping(nested.get("safety"), "nested_inventory_safety_invalid")
    if safety != {
        "formal_training_authorized": False,
        "gpu_requests": 0,
        "metadata_only": True,
        "model_loads": 0,
        "network_requests": 0,
        "provider_requests": 0,
        "sample_content_emitted": False,
    }:
        _fail(f"inventory_{source_class}_safety_drift")
    extraction = _mapping(nested.get("extraction"), "nested_extraction_invalid")
    if extraction.get("body_files_read") != 0:
        _fail(f"inventory_{source_class}_body_read_drift")

    if binding.status == "unavailable":
        forbidden = ("source_id_count", "source_id_inventory_sha256", "inventory_file")
        if any(field in nested or field in entry for field in forbidden):
            _fail(f"inventory_{source_class}_unavailable_identity_minted")
        expected_missing = {
            "source_id_count",
            "source_id_inventory_sha256",
            "inventory_file",
        }
        if set(nested.get("missing_fields", ())) != expected_missing:
            _fail(f"inventory_{source_class}_missing_fields_drift")
        return 0

    if (
        entry.get("source_id_count") != binding.source_id_count
        or entry.get("source_id_inventory_sha256") != binding.source_id_inventory_sha256
    ):
        _fail(f"inventory_{source_class}_main_ready_identity_drift")
    if (
        nested.get("source_id_count") != binding.source_id_count
        or nested.get("source_id_inventory_sha256")
        != binding.source_id_inventory_sha256
    ):
        _fail(f"inventory_{source_class}_nested_ready_identity_drift")
    inventory_file = _mapping(
        nested.get("inventory_file"), f"inventory_{source_class}_file_invalid"
    )
    expected_file_path = f"inventories/{source_class}/source_ids.sha256.jsonl"
    if (
        inventory_file.get("path") != expected_file_path
        or inventory_file.get("sha256") != binding.source_ids_file_sha256
        or inventory_file.get("records") != (binding.source_id_count)
    ):
        _fail(f"inventory_{source_class}_file_binding_drift")
    source_ids_snapshot = ledger.read(
        _inventory_relative(source_class, "source_ids.sha256.jsonl"),
        f"inventory_{source_class}_source_ids_unreadable",
    )
    if (
        binding.source_ids_file_sha256 is None
        or binding.source_id_count is None
        or binding.source_id_inventory_sha256 is None
    ):
        _fail(f"inventory_{source_class}_ready_binding_incomplete")
    _validate_hash_inventory(
        source_ids_snapshot,
        expected_file_sha256=binding.source_ids_file_sha256,
        expected_count=binding.source_id_count,
        expected_logical_sha256=binding.source_id_inventory_sha256,
        code=f"inventory_{source_class}_source_ids",
    )
    if inventory_file.get("bytes") != len(source_ids_snapshot.data):
        _fail(f"inventory_{source_class}_file_bytes_drift")
    return 1


def evaluate_toy_prerequisite(config_path: str | Path) -> dict[str, Any]:
    """Authenticate metadata and return an irrevocably blocked decision."""

    requested = Path(config_path)
    if requested.is_absolute():
        try:
            config_relative = requested.resolve().relative_to(
                _REPOSITORY_ROOT.resolve()
            )
        except ValueError:
            _fail("consumer_config_path_invalid")
        config_relative_text = config_relative.as_posix()
    else:
        if ".." in requested.parts:
            _fail("consumer_config_path_invalid")
        config_relative_text = requested.as_posix()
    if config_relative_text != READ_WHITELIST[0]:
        _fail("consumer_config_path_invalid")

    ledger = _ReadLedger(_REPOSITORY_ROOT)
    config_snapshot = ledger.read(config_relative_text, "consumer_config_unreadable")
    if config_snapshot.sha256 != CONFIG_SHA256:
        _fail("consumer_config_sha256_mismatch")
    config = _decode_yaml(config_snapshot, "consumer_config_invalid")
    _validate_consumer_config(config)
    paths = _mapping(config["paths"], "consumer_paths_invalid")

    inventory_schema = ledger.read(
        str(paths["protected_inventory_schema"]), "inventory_schema_unreadable"
    )
    record_schema = ledger.read(
        str(paths["toy_record_schema"]), "record_schema_unreadable"
    )
    main_schema = ledger.read(
        str(paths["main_manifest_schema"]), "main_schema_unreadable"
    )
    trigger_schema = ledger.read(
        str(paths["trigger_receipt_schema"]), "trigger_schema_unreadable"
    )
    producer_config_snapshot = ledger.read(
        str(paths["producer_config"]), "producer_config_unreadable"
    )
    grammar_snapshot = ledger.read(
        str(paths["closed_grammar"]), "closed_grammar_unreadable"
    )
    for snapshot, expected, code in (
        (inventory_schema, INVENTORY_SCHEMA_SHA256, "inventory_schema"),
        (record_schema, RECORD_SCHEMA_SHA256, "record_schema"),
        (main_schema, MAIN_SCHEMA_SHA256, "main_schema"),
        (trigger_schema, TRIGGER_SCHEMA_SHA256, "trigger_schema"),
    ):
        _validate_schema_document(snapshot, expected, code)
    if producer_config_snapshot.sha256 != PRODUCER_CONFIG_SHA256:
        _fail("producer_config_sha256_mismatch")
    producer_config = _decode_json(producer_config_snapshot, "producer_config_invalid")
    _validate_producer_config(producer_config)
    if grammar_snapshot.sha256 != CLOSED_GRAMMAR_SHA256:
        _fail("closed_grammar_sha256_mismatch")
    grammar = _decode_json(grammar_snapshot, "closed_grammar_invalid")
    grammar_safety = _mapping(grammar.get("safety"), "closed_grammar_safety_invalid")
    if grammar.get("schema_version") != "anchor.qwen-toy-closed-grammar.v1" or (
        grammar_safety.get("formal_training_authorized") is not False
    ):
        _fail("closed_grammar_contract_drift")

    main_snapshot = ledger.read(str(paths["main_manifest"]), "main_manifest_unreadable")
    main_sidecar = ledger.read(
        f"{paths['main_manifest']}.sha256", "main_sidecar_unreadable"
    )
    if main_snapshot.sha256 != MAIN_MANIFEST_SHA256:
        _fail("main_manifest_sha256_mismatch")
    _validate_sidecar(
        main_sidecar,
        target_sha256=MAIN_MANIFEST_SHA256,
        target_name="manifest.json",
        physical_sha256=MAIN_SIDECAR_SHA256,
        code="main_sidecar",
    )
    main = _decode_json(main_snapshot, "main_manifest_invalid")
    _validate_instance(main_schema, MAIN_SCHEMA_SHA256, main, "main_manifest")
    _validate_main_semantics(
        main, producer_config=producer_config, trigger_schema=trigger_schema
    )

    audit_snapshot = ledger.read(str(paths["audit"]), "audit_unreadable")
    audit_sidecar = ledger.read(f"{paths['audit']}.sha256", "audit_sidecar_unreadable")
    if audit_snapshot.sha256 != AUDIT_SHA256:
        _fail("audit_sha256_mismatch")
    _validate_sidecar(
        audit_sidecar,
        target_sha256=AUDIT_SHA256,
        target_name="audit.json",
        physical_sha256=AUDIT_SIDECAR_SHA256,
        code="audit_sidecar",
    )
    audit = _decode_json(audit_snapshot, "audit_invalid")
    _validate_audit(main, audit)

    toy_source_ids = ledger.read(
        "fixtures/research/qwen_toy_prerequisite_v1/toy/source_ids.sha256.jsonl",
        "toy_source_ids_unreadable",
    )
    _validate_hash_inventory(
        toy_source_ids,
        expected_file_sha256=TOY_SOURCE_IDS_FILE_SHA256,
        expected_count=8,
        expected_logical_sha256=TOY_SOURCE_ID_INVENTORY_SHA256,
        code="toy_source_ids",
    )

    inventory_entries = main.get("protected_inventories", ())
    if not isinstance(inventory_entries, list) or len(inventory_entries) != 6:
        _fail("protected_inventory_count_drift")
    ready_verified = 0
    for source_class, raw_entry in zip(ORDERED_SOURCE_CLASSES, inventory_entries):
        entry = _mapping(raw_entry, "protected_inventory_entry_invalid")
        binding = INVENTORY_BINDINGS[source_class]
        manifest_snapshot = ledger.read(
            _inventory_relative(source_class, "manifest.json"),
            f"inventory_{source_class}_manifest_unreadable",
        )
        sidecar_snapshot = ledger.read(
            _inventory_relative(source_class, "manifest.json.sha256"),
            f"inventory_{source_class}_sidecar_unreadable",
        )
        if manifest_snapshot.sha256 != binding.manifest_sha256:
            _fail(f"inventory_{source_class}_manifest_sha256_mismatch")
        _validate_sidecar(
            sidecar_snapshot,
            target_sha256=binding.manifest_sha256,
            target_name="manifest.json",
            physical_sha256=binding.sidecar_sha256,
            code=f"inventory_{source_class}_sidecar",
        )
        nested = _decode_json(
            manifest_snapshot, f"inventory_{source_class}_manifest_invalid"
        )
        ready_verified += _validate_inventory_entry(
            source_class=source_class,
            entry=entry,
            nested=nested,
            binding=binding,
            inventory_schema=inventory_schema,
            ledger=ledger,
        )
    if ready_verified != 2:
        _fail("ready_inventory_count_drift")
    if set(ledger.snapshots) != set(READ_WHITELIST):
        _fail("read_whitelist_not_exactly_consumed")

    ledger.final_recheck()
    return {
        "schema_version": "anchor.qwen-toy-prerequisite-consumer-decision.v1",
        "status": "blocked",
        "diagnostic_metadata_verified": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "zero_intersection_claimed": False,
        "v1_attestation_emitted": False,
        "reason": "protected_inventory_incomplete_and_trigger_receipt_pending",
        "producer_commit": PRODUCER_COMMIT,
        "protected_inventory_coverage": {
            "ready": 2,
            "total": 6,
            "ready_source_classes": list(READY_SOURCE_CLASSES),
            "unavailable_source_classes": list(UNAVAILABLE_SOURCE_CLASSES),
        },
        "trigger_receipt": {
            "status": "pending_request_local_materialization",
            "bound_identity_count": 0,
            "token_ids_emitted": False,
            "planner_request1_private_kv_reused": False,
        },
        "audit": {
            "unique_metadata_files_read": len(ledger.snapshots),
            "hash_id_inventories_verified": 3,
            "toy_record_body_read": False,
            "protected_content_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate the metadata-only Qwen toy prerequisite"
    )
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = evaluate_toy_prerequisite(args.config)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
