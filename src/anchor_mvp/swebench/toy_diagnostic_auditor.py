"""Independent auditor for the deterministic diagnostic toy partition.

This module intentionally does not import the toy generator.  It rebuilds the
small closed-grammar partition from authenticated config and grammar values so
that changing a record together with its manifest is still detected.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any


AUDITOR_VERSION = "anchor.qwen-toy-diagnostic-attester.v1"
GENERATOR_VERSION = "anchor.qwen-toy-diagnostic-generator.v1"
RECORD_SCHEMA = "anchor.qwen-toy-diagnostic-record.v1"
GRAMMAR_SCHEMA = "anchor.qwen-toy-closed-grammar.v1"
TOY_NAMESPACE = "anchor.qwen-toy-diagnostic.v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]{2,255}$")
_ATOM_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


class ToyDiagnosticAuditError(ValueError):
    """A stable, content-free audit failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ToyDiagnosticAuditError(code)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _select(seed: str, index: int, slot: str, upper: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{index}\0{slot}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % upper


def _source_id_token(namespace: str, native_identifier: str) -> str:
    if _ID_RE.fullmatch(namespace) is None or not native_identifier:
        _fail("toy_audit_source_id_invalid")
    return _sha256_bytes(f"{namespace}\0{native_identifier}".encode())


def _inventory_sha256(tokens: list[str]) -> str:
    ordered = sorted(set(tokens))
    if len(ordered) != len(tokens) or any(
        re.fullmatch(r"[0-9a-f]{64}", item) is None for item in ordered
    ):
        _fail("toy_audit_inventory_invalid")
    return _sha256_bytes("\n".join(ordered).encode("ascii"))


def _rebuild(
    config_value: object, grammar_value: object
) -> tuple[list[dict[str, Any]], list[str]]:
    config = _mapping(config_value, "toy_audit_config_invalid")
    _exact_keys(
        config,
        {
            "schema_version",
            "generator_version",
            "toy_namespace",
            "partition",
            "deterministic_seed",
            "record_count",
            "output_file",
            "source_id_token_algorithm",
            "source_id_inventory_algorithm",
        },
        "toy_audit_config_invalid",
    )
    if (
        config["schema_version"] != "anchor.qwen-toy-generator-config.v1"
        or config["generator_version"] != GENERATOR_VERSION
        or config["toy_namespace"] != TOY_NAMESPACE
        or config["partition"] != "diagnostic_only"
        or config["output_file"] != "toy/diagnostic.jsonl"
        or config["source_id_token_algorithm"]
        != "sha256_utf8_namespace_nul_native_identifier_v1"
        or config["source_id_inventory_algorithm"]
        != "sha256_sorted_unique_hex_lines_no_trailing_lf_v1"
        or not isinstance(config["deterministic_seed"], str)
        or not config["deterministic_seed"]
        or not isinstance(config["record_count"], int)
        or isinstance(config["record_count"], bool)
        or not 1 <= config["record_count"] <= 128
    ):
        _fail("toy_audit_config_invalid")

    grammar = _mapping(grammar_value, "toy_audit_grammar_invalid")
    _exact_keys(
        grammar,
        {"schema_version", "grammar_id", "atoms", "operations", "safety"},
        "toy_audit_grammar_invalid",
    )
    atoms = grammar["atoms"]
    operations = grammar["operations"]
    safety = _mapping(grammar["safety"], "toy_audit_grammar_invalid")
    if (
        grammar["schema_version"] != GRAMMAR_SCHEMA
        or grammar["grammar_id"] != "anchor.qwen-toy-closed-grammar.2026-07.v1"
        or not isinstance(atoms, list)
        or not 4 <= len(atoms) <= 64
        or len(set(atoms)) != len(atoms)
        or any(
            not isinstance(item, str) or _ATOM_RE.fullmatch(item) is None
            for item in atoms
        )
        or not isinstance(operations, list)
        or not 1 <= len(operations) <= 8
        or safety
        != {
            "protected_sources_used": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "formal_training_authorized": False,
        }
    ):
        _fail("toy_audit_grammar_invalid")

    checked_operations: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in operations:
        operation = _mapping(raw, "toy_audit_grammar_invalid")
        _exact_keys(operation, {"id", "kind", "separator"}, "toy_audit_grammar_invalid")
        if (
            not isinstance(operation["id"], str)
            or _ID_RE.fullmatch(operation["id"]) is None
            or operation["id"] in seen
            or operation["kind"] not in {"join", "reverse_join", "sorted_join"}
            or operation["separator"] not in {"|", ":", "/"}
        ):
            _fail("toy_audit_grammar_invalid")
        seen.add(operation["id"])
        checked_operations.append(operation)

    seed = str(config["deterministic_seed"])
    records: list[dict[str, Any]] = []
    tokens: list[str] = []
    for index in range(int(config["record_count"])):
        left_index = _select(seed, index, "left", len(atoms))
        right_index = _select(seed, index, "right", len(atoms))
        if right_index == left_index:
            right_index = (right_index + 1) % len(atoms)
        operation = checked_operations[
            _select(seed, index, "operation", len(checked_operations))
        ]
        left = atoms[left_index]
        right = atoms[right_index]
        separator = str(operation["separator"])
        if operation["kind"] == "join":
            result = f"{left}{separator}{right}"
        elif operation["kind"] == "reverse_join":
            result = f"{right}{separator}{left}"
        else:
            result = separator.join(sorted((left, right)))
        identity_digest = _sha256_bytes(
            f"{seed}\0{index}\0{operation['id']}\0{left}\0{right}".encode()
        )
        source_id = f"{TOY_NAMESPACE}:{index:04d}:{identity_digest[:16]}"
        token = _source_id_token(TOY_NAMESPACE, source_id)
        records.append(
            {
                "schema_version": RECORD_SCHEMA,
                "source_id": source_id,
                "source_id_token_sha256": token,
                "partition": "diagnostic_only",
                "operation": operation["id"],
                "input": {"left": left, "right": right},
                "target": {"value": result},
                "formal_training_authorized": False,
            }
        )
        tokens.append(token)
    return records, tokens


def audit_toy_partition(
    config_value: object,
    grammar_value: object,
    record_bytes: bytes,
    inventory_bytes: bytes,
) -> dict[str, object]:
    """Rebuild and authenticate one canonical JSONL toy partition."""

    expected_records, expected_tokens = _rebuild(config_value, grammar_value)
    expected_record_bytes = b"".join(
        _canonical_bytes(record) + b"\n" for record in expected_records
    )
    expected_inventory_bytes = b"".join(
        token.encode("ascii") + b"\n" for token in sorted(expected_tokens)
    )
    if record_bytes != expected_record_bytes:
        _fail("toy_audit_record_rebuild_mismatch")
    if inventory_bytes != expected_inventory_bytes:
        _fail("toy_audit_inventory_rebuild_mismatch")

    reparsed: list[object] = []
    for raw_line in record_bytes.splitlines(keepends=True):
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            _fail("toy_audit_jsonl_noncanonical")
        try:
            value = json.loads(raw_line[:-1].decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToyDiagnosticAuditError("toy_audit_jsonl_invalid") from exc
        if _canonical_bytes(value) + b"\n" != raw_line:
            _fail("toy_audit_jsonl_noncanonical")
        reparsed.append(value)
    if reparsed != expected_records:
        _fail("toy_audit_reparse_mismatch")

    return {
        "schema_version": "anchor.qwen-toy-diagnostic-audit.v1",
        "status": "passed",
        "auditor_version": AUDITOR_VERSION,
        "record_count": len(expected_records),
        "records_sha256": _sha256_bytes(record_bytes),
        "source_id_count": len(expected_tokens),
        "source_id_inventory_sha256": _inventory_sha256(expected_tokens),
        "source_id_inventory_file_sha256": _sha256_bytes(inventory_bytes),
        "independent_rebuild_passed": True,
        "same_snapshot_reparse_passed": True,
        "formal_training_authorized": False,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_content_reads": 0,
    }
