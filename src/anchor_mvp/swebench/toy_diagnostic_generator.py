"""Deterministic diagnostic-only toy data generator.

The generator is deliberately independent from SWE-bench, Gold, held-out, and
scaffold inputs.  Its complete semantic read set is its own implementation,
one versioned configuration, and one closed grammar.  It never loads a model,
uses a provider, or grants training authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any


GENERATOR_VERSION = "anchor.qwen-toy-diagnostic-generator.v1"
RECORD_SCHEMA = "anchor.qwen-toy-diagnostic-record.v1"
GRAMMAR_SCHEMA = "anchor.qwen-toy-closed-grammar.v1"
PARTITION = "diagnostic_only"
TOY_NAMESPACE = "anchor.qwen-toy-diagnostic.v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]{2,255}$")
_ATOM_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


class ToyDiagnosticGeneratorError(ValueError):
    """A stable, content-free generator failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ToyDiagnosticGeneratorError(code)


def canonical_bytes(value: object) -> bytes:
    """Encode the repository's canonical compact JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def source_id_token(namespace: str, native_identifier: str) -> str:
    """Hash one identifier under an explicit namespace and domain separator."""

    if _ID_RE.fullmatch(namespace) is None or not native_identifier:
        _fail("toy_generator_source_id_invalid")
    return sha256_bytes(f"{namespace}\0{native_identifier}".encode("utf-8"))


def source_id_inventory_sha256(tokens: Sequence[str]) -> str:
    """Hash sorted unique opaque source-ID tokens without a trailing newline."""

    ordered = sorted(set(tokens))
    if len(ordered) != len(tokens) or any(
        re.fullmatch(r"[0-9a-f]{64}", item) is None for item in ordered
    ):
        _fail("toy_generator_source_id_inventory_invalid")
    return sha256_bytes("\n".join(ordered).encode("ascii"))


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _select(seed: str, index: int, slot: str, upper: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{index}\0{slot}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % upper


def _validate_generator_config(value: object) -> Mapping[str, Any]:
    config = _mapping(value, "toy_generator_config_invalid")
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
        "toy_generator_config_invalid",
    )
    if (
        config["schema_version"] != "anchor.qwen-toy-generator-config.v1"
        or config["generator_version"] != GENERATOR_VERSION
        or config["toy_namespace"] != TOY_NAMESPACE
        or config["partition"] != PARTITION
        or config["source_id_token_algorithm"]
        != "sha256_utf8_namespace_nul_native_identifier_v1"
        or config["source_id_inventory_algorithm"]
        != "sha256_sorted_unique_hex_lines_no_trailing_lf_v1"
        or not isinstance(config["deterministic_seed"], str)
        or not config["deterministic_seed"]
        or not isinstance(config["record_count"], int)
        or isinstance(config["record_count"], bool)
        or not 1 <= config["record_count"] <= 128
        or config["output_file"] != "toy/diagnostic.jsonl"
    ):
        _fail("toy_generator_config_invalid")
    return config


def _validate_grammar(
    value: object,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    grammar = _mapping(value, "toy_generator_grammar_invalid")
    _exact_keys(
        grammar,
        {"schema_version", "grammar_id", "atoms", "operations", "safety"},
        "toy_generator_grammar_invalid",
    )
    atoms = grammar["atoms"]
    operations = grammar["operations"]
    safety = _mapping(grammar["safety"], "toy_generator_grammar_invalid")
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
    ):
        _fail("toy_generator_grammar_invalid")
    _exact_keys(
        safety,
        {
            "protected_sources_used",
            "provider_requests",
            "network_requests",
            "model_loads",
            "gpu_requests",
            "formal_training_authorized",
        },
        "toy_generator_grammar_invalid",
    )
    if safety != {
        "protected_sources_used": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "formal_training_authorized": False,
    }:
        _fail("toy_generator_grammar_invalid")
    checked_operations: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in operations:
        operation = _mapping(item, "toy_generator_grammar_invalid")
        _exact_keys(
            operation,
            {"id", "kind", "separator"},
            "toy_generator_grammar_invalid",
        )
        if (
            not isinstance(operation["id"], str)
            or _ID_RE.fullmatch(operation["id"]) is None
            or operation["id"] in seen
            or operation["kind"] not in {"join", "reverse_join", "sorted_join"}
            or operation["separator"] not in {"|", ":", "/"}
        ):
            _fail("toy_generator_grammar_invalid")
        seen.add(operation["id"])
        checked_operations.append(operation)
    return tuple(atoms), tuple(checked_operations)


def generate_toy_records(
    config_value: object,
    grammar_value: object,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate records and opaque namespaced source-ID tokens."""

    config = _validate_generator_config(config_value)
    atoms, operations = _validate_grammar(grammar_value)
    seed = str(config["deterministic_seed"])
    records: list[dict[str, Any]] = []
    tokens: list[str] = []
    for index in range(int(config["record_count"])):
        left_index = _select(seed, index, "left", len(atoms))
        right_index = _select(seed, index, "right", len(atoms))
        if right_index == left_index:
            right_index = (right_index + 1) % len(atoms)
        operation = operations[_select(seed, index, "operation", len(operations))]
        left = atoms[left_index]
        right = atoms[right_index]
        separator = str(operation["separator"])
        if operation["kind"] == "join":
            result = f"{left}{separator}{right}"
        elif operation["kind"] == "reverse_join":
            result = f"{right}{separator}{left}"
        else:
            result = separator.join(sorted((left, right)))
        identity_digest = sha256_bytes(
            f"{seed}\0{index}\0{operation['id']}\0{left}\0{right}".encode("utf-8")
        )
        source_id = f"{TOY_NAMESPACE}:{index:04d}:{identity_digest[:16]}"
        token = source_id_token(TOY_NAMESPACE, source_id)
        records.append(
            {
                "schema_version": RECORD_SCHEMA,
                "source_id": source_id,
                "source_id_token_sha256": token,
                "partition": PARTITION,
                "operation": operation["id"],
                "input": {"left": left, "right": right},
                "target": {"value": result},
                "formal_training_authorized": False,
            }
        )
        tokens.append(token)
    if len({row["source_id"] for row in records}) != len(records):
        _fail("toy_generator_duplicate_source_id")
    source_id_inventory_sha256(tokens)
    return records, tokens
